from collections import OrderedDict
from collections.abc import Generator, Iterable, Sequence
from logging import getLogger
from typing import Optional
import os
import requests
from datetime import datetime, timezone

from django.conf import settings
from django.db import transaction

from eth_typing import ChecksumAddress, HexStr
from hexbytes import HexBytes
from safe_eth.eth import EthereumClient
from safe_eth.util.util import to_0x_hex_str
from web3.types import BlockTrace, FilterTrace

from safe_transaction_service.contracts.tx_decoder import (
    CannotDecode,
    UnexpectedProblemDecoding,
    get_safe_tx_decoder,
)
from safe_transaction_service.utils.utils import chunks

from ..models import (
    InternalTx,
    InternalTxDecoded,
    SafeMasterCopy,
    SafeRelevantTransaction,
    SafeContract,
    EthereumBlock,
    EthereumTx,
)
from .element_already_processed_checker import ElementAlreadyProcessedChecker
from .ethereum_indexer import EthereumIndexer, FindRelevantElementsException

logger = getLogger(__name__)

# Xone network configuration
XONE_CHAIN_ID = 3721
XONE_EXPLORER_APIS = [
    "https://xscscan.com/api/v2",
    "https://api.xonescan.com/api"
]


def _try_explorer_apis(endpoint_path: str, params: dict = None, timeout: int = 10) -> Optional[requests.Response]:
    """Try all Explorer APIs and return the first successful response"""
    for api in XONE_EXPLORER_APIS:
        try:
            url = f"{api}{endpoint_path}"
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.debug(f"Explorer API {api} failed: {e}")
            continue
    return None


class InternalTxIndexerProvider:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = cls.get_new_instance()
        return cls.instance

    @classmethod
    def get_new_instance(cls) -> "InternalTxIndexer":
        from django.conf import settings

        # Check if this is Xone network
        ethereum_client = EthereumClient(settings.ETHEREUM_NODE_URL)
        try:
            chain_id = ethereum_client.w3.eth.chain_id
            if chain_id == XONE_CHAIN_ID:
                logger.info(f"Detected Xone network (chain_id={XONE_CHAIN_ID}), using XoneInternalTxIndexer")
                return XoneInternalTxIndexer(ethereum_client)
        except Exception as e:
            logger.debug(f"Failed to get chain_id: {e}")

        if settings.ETH_INTERNAL_NO_FILTER:
            instance_class = InternalTxIndexerWithTraceBlock
        else:
            instance_class = InternalTxIndexer

        return instance_class(
            EthereumClient(settings.ETHEREUM_TRACING_NODE_URL),
        )

    @classmethod
    def del_singleton(cls):
        if hasattr(cls, "instance"):
            del cls.instance


class InternalTxIndexer(EthereumIndexer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "block_process_limit", settings.ETH_INTERNAL_TXS_BLOCK_PROCESS_LIMIT
        )
        kwargs.setdefault(
            "block_process_limit_max", settings.ETH_INTERNAL_TXS_BLOCK_PROCESS_LIMIT_MAX
        )
        kwargs.setdefault(
            "blocks_to_reindex_again", settings.ETH_INTERNAL_TXS_BLOCKS_TO_REINDEX_AGAIN
        )
        super().__init__(*args, **kwargs)

        self.trace_txs_batch_size: int = settings.ETH_INTERNAL_TRACE_TXS_BATCH_SIZE
        self.number_trace_blocks: int = settings.ETH_INTERNAL_TXS_NUMBER_TRACE_BLOCKS
        self.tx_decoder = get_safe_tx_decoder()
        self.element_already_processed_checker = ElementAlreadyProcessedChecker()

    @property
    def database_field(self):
        return "tx_block_number"

    @property
    def database_queryset(self):
        return SafeMasterCopy.objects.all()

    def find_relevant_elements(
        self,
        addresses: set[ChecksumAddress],
        from_block_number: int,
        to_block_number: int,
        current_block_number: int | None = None,
    ) -> OrderedDict[bytes, None | list[BlockTrace]]:
        current_block_number = (
            current_block_number or self.ethereum_client.current_block_number
        )
        # Use `trace_block` for last `number_trace_blocks` blocks and `trace_filter` for the others
        trace_block_number = max(current_block_number - self.number_trace_blocks, 0)
        if from_block_number > trace_block_number:  # Just trace_block
            return self._find_relevant_elements_using_trace_block(
                addresses, from_block_number, to_block_number
            )
        elif to_block_number < trace_block_number:  # Just trace_filter
            return self._find_relevant_elements_using_trace_filter(
                addresses, from_block_number, to_block_number
            )
        else:  # trace_filter for old blocks and trace_block for the most recent ones
            logger.debug(
                "Using trace_filter from-block=%d to-block=%d and trace_block from-block=%d to-block=%d",
                from_block_number,
                trace_block_number,
                trace_block_number,
                to_block_number,
            )
            relevant_elements = self._find_relevant_elements_using_trace_filter(
                addresses, from_block_number, trace_block_number
            )
            relevant_elements.update(
                self._find_relevant_elements_using_trace_block(
                    addresses, trace_block_number + 1, to_block_number
                )
            )
            return relevant_elements

    def _find_relevant_elements_using_trace_block(
        self,
        addresses: set[ChecksumAddress],
        from_block_number: int,
        to_block_number: int,
    ) -> OrderedDict[bytes, list[BlockTrace]]:
        addresses_set = set(addresses)  # More optimal to use with `in`
        logger.debug(
            "Using trace_block from-block=%d to-block=%d",
            from_block_number,
            to_block_number,
        )
        try:
            block_numbers = list(range(from_block_number, to_block_number + 1))

            with self.auto_adjust_block_limit(from_block_number, to_block_number):
                all_blocks_traces = self.ethereum_client.tracing.trace_blocks(
                    block_numbers
                )
            traces: OrderedDict[bytes, list[BlockTrace]] = OrderedDict()
            relevant_tx_hashes: set[bytes] = set()
            for block_number, block_traces in zip(
                block_numbers, all_blocks_traces, strict=False
            ):
                if not block_traces:
                    logger.warning("Empty `trace_block` for block=%d", block_number)

                for trace in block_traces:
                    transaction_hash: bytes = trace.get("transactionHash")
                    if transaction_hash:
                        traces.setdefault(transaction_hash, []).append(trace)
                        # We're only interested in traces related to the provided addresses
                        if (
                            trace.get("action", {}).get("from") in addresses_set
                            or trace.get("action", {}).get("to") in addresses_set
                        ):
                            relevant_tx_hashes.add(transaction_hash)

            # Remove not relevant traces
            for tx_hash in list(traces.keys()):
                if tx_hash not in relevant_tx_hashes:
                    del traces[tx_hash]

            return traces
        except OSError as e:
            raise FindRelevantElementsException(
                "Request error calling `trace_block`"
            ) from e

    def _find_relevant_elements_using_trace_filter(
        self,
        addresses: set[ChecksumAddress],
        from_block_number: int,
        to_block_number: int,
    ) -> OrderedDict[bytes, None]:
        """
        Search for tx hashes with internal txs (in and out) of a `address`

        :param addresses:
        :param from_block_number: Starting block number
        :param to_block_number: Ending block number
        :return: Tx hashes of txs with internal txs relevant for the `addresses`
        """
        logger.debug(
            "Using trace_filter from-block=%d to-block=%d",
            from_block_number,
            to_block_number,
        )

        try:
            # We only need to search for traces `to` the provided addresses
            with self.auto_adjust_block_limit(from_block_number, to_block_number):
                to_traces = self.ethereum_client.tracing.trace_filter(
                    from_block=from_block_number,
                    to_block=to_block_number,
                    to_address=list(addresses),
                )
        except OSError as e:
            raise FindRelevantElementsException(
                "Request error calling `trace_filter`"
            ) from e
        except ValueError as e:
            # For example, Infura returns:
            #   ValueError: {'code': -32005, 'data': {'from': '0x6BBCE1', 'limit': 10000, 'to': '0x7072DB'}, 'message': 'query returned more than 10000 results. Try with this block range [0x6BBCE1, 0x7072DB].'}
            logger.warning(
                "%s: Value error retrieving trace_filter results from-block=%d to-block=%d : %s",
                self.__class__.__name__,
                from_block_number,
                to_block_number,
                e,
            )
            raise FindRelevantElementsException(
                f"Request error retrieving trace_filter results "
                f"from-block={from_block_number} to-block={to_block_number}"
            ) from e

        # Log INFO if traces found, DEBUG if not
        traces: OrderedDict[bytes, None] = OrderedDict()
        for trace in to_traces:
            transaction_hash = trace.get("transactionHash")
            if transaction_hash:
                # Leave this empty, as we are missing traces for the transaction and will need to be fetched later
                traces[transaction_hash] = None

        log_fn = logger.info if traces else logger.debug
        log_fn(
            "Found %d relevant txs with internal txs between block-number=%d and block-number=%d. Addresses=%s",
            len(traces),
            from_block_number,
            to_block_number,
            addresses,
        )

        return traces

    def _get_internal_txs_to_decode(
        self, tx_hashes: Sequence[str]
    ) -> Generator[InternalTxDecoded]:
        """
        Retrieve relevant `InternalTxs` and if possible decode them to return `InternalTxsDecoded`

        :return: A `InternalTxDecoded` generator to be more RAM friendly
        """
        for internal_tx in (
            InternalTx.objects.can_be_decoded()
            .filter(ethereum_tx__in=tx_hashes)
            .iterator()
        ):
            data = bytes(internal_tx.data)
            try:
                function_name, arguments = self.tx_decoder.decode_transaction(data)
                yield InternalTxDecoded(
                    internal_tx=internal_tx,
                    function_name=function_name,
                    arguments=arguments,
                    processed=False,
                )
            except CannotDecode as exc:
                logger.debug("Cannot decode %s: %s", to_0x_hex_str(data), exc)
            except UnexpectedProblemDecoding as exc:
                logger.warning(
                    "Unexpected problem decoding %s: %s", to_0x_hex_str(data), exc
                )

    def trace_transactions(
        self, tx_hashes: Sequence[HexStr], batch_size: int
    ) -> Iterable[list[FilterTrace]]:
        batch_size = batch_size or len(tx_hashes)  # If `0`, don't use batches
        for tx_hash_chunk in chunks(list(tx_hashes), batch_size):
            tx_hash_chunk = list(tx_hash_chunk)
            try:
                yield from self.ethereum_client.tracing.trace_transactions(
                    tx_hash_chunk
                )
            except OSError:
                logger.error(
                    "Problem calling `trace_transactions` with %d txs. "
                    "Try lowering ETH_INTERNAL_TRACE_TXS_BATCH_SIZE",
                    len(tx_hash_chunk),
                    exc_info=True,
                )
                raise

    def filter_relevant_txs(
        self, internal_txs: Generator[InternalTx]
    ) -> Generator[InternalTx]:
        for internal_tx in internal_txs:
            if internal_tx.is_relevant:
                if internal_tx.is_ether_transfer:
                    SafeRelevantTransaction.objects.get_or_create(
                        ethereum_tx_id=internal_tx.ethereum_tx_id,
                        safe=internal_tx.to,
                        defaults={
                            "timestamp": internal_tx.timestamp,
                        },
                    )
                yield internal_tx

    def process_elements(
        self, tx_hash_with_traces: OrderedDict[bytes, FilterTrace | None]
    ) -> list[HexBytes]:
        """
        :param tx_hash_with_traces:
        :return: Inserted `InternalTx` objects
        """
        if not tx_hash_with_traces:
            return []

        # Copy as we might modify it
        tx_hash_with_traces = dict(tx_hash_with_traces)

        logger.debug(
            "Prefetching and storing %d ethereum txs", len(tx_hash_with_traces)
        )

        tx_hashes = []
        tx_hashes_missing_traces = []
        for tx_hash in list(tx_hash_with_traces.keys()):
            # Check if transactions have already been processed
            # Provide block_hash if available as a mean to prevent reorgs
            block_hash = (
                tx_hash_with_traces[tx_hash][0]["blockHash"]
                if tx_hash_with_traces[tx_hash]
                else None
            )
            if not self.element_already_processed_checker.is_processed(
                tx_hash, block_hash
            ):
                tx_hashes.append(tx_hash)
                # Traces can be already populated if using `trace_block`, but with `trace_filter`
                # some traces will be missing and `trace_transaction` needs to be called
                if not tx_hash_with_traces[tx_hash]:
                    tx_hashes_missing_traces.append(tx_hash)
            else:
                # Traces were already processed
                del tx_hash_with_traces[tx_hash]

        ethereum_txs = self.index_service.txs_create_or_update_from_tx_hashes(tx_hashes)
        logger.debug("End prefetching and storing of ethereum txs")

        logger.debug("Prefetching of traces(internal txs)")
        missing_traces_lists = self.trace_transactions(
            tx_hashes_missing_traces, batch_size=self.trace_txs_batch_size
        )
        for tx_hash_missing_traces, missing_traces in zip(
            tx_hashes_missing_traces, missing_traces_lists, strict=False
        ):
            tx_hash_with_traces[tx_hash_missing_traces] = missing_traces

        internal_txs = (
            InternalTx.objects.build_from_trace(trace, ethereum_tx)
            for ethereum_tx in ethereum_txs
            for trace in self.ethereum_client.tracing.filter_out_errored_traces(
                tx_hash_with_traces[HexBytes(ethereum_tx.tx_hash)]
            )
        )

        logger.debug("End prefetching of traces(internal txs)")

        with transaction.atomic():
            logger.debug("Storing traces")
            traces_stored = InternalTx.objects.bulk_create_from_generator(
                self.filter_relevant_txs(internal_txs), ignore_conflicts=True
            )
            logger.debug("Stored %d traces", traces_stored)

            logger.debug("Start decoding and storing of decoded traces")
            #  Pass `tx_hashes` instead of `InternalTxs` to `_get_internal_txs_to_decode`
            #  as they must be retrieved again.
            #  `bulk_create` with `ignore_conflicts=True` do not populate the `pk` when storing objects
            internal_txs_decoded = InternalTxDecoded.objects.bulk_create_from_generator(
                self._get_internal_txs_to_decode(tx_hashes), ignore_conflicts=True
            )
            logger.debug(
                "End decoding and storing of %d decoded traces", internal_txs_decoded
            )

        # Mark traces as processed
        for tx_hash in list(tx_hash_with_traces.keys()):
            block_hash = (
                tx_hash_with_traces[tx_hash][0]["blockHash"]
                if tx_hash_with_traces[tx_hash]
                else None
            )
            self.element_already_processed_checker.mark_as_processed(
                tx_hash, block_hash
            )
        return tx_hashes


class InternalTxIndexerWithTraceBlock(InternalTxIndexer):
    """
    Indexer for nodes not supporting `trace_filter`, so it will always use `trace_block`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_almost_updated_addresses(
        self, current_block_number: int
    ) -> set[ChecksumAddress]:
        """
        Return every address. As we are using `trace_block` every master copy should be processed together

        :param current_block_number:
        :return:
        """
        return self.get_not_updated_addresses(current_block_number)

    def find_relevant_elements(
        self,
        addresses: set[ChecksumAddress],
        from_block_number: int,
        to_block_number: int,
        current_block_number: int | None = None,
    ) -> OrderedDict[bytes, FilterTrace]:
        return self._find_relevant_elements_using_trace_block(
            addresses, from_block_number, to_block_number
        )


class XoneInternalTxIndexer(InternalTxIndexer):
    """
    Xone network specific indexer - uses Explorer API instead of trace API
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safe_addresses = set()
        self.use_explorer_api = getattr(settings, 'XONE_USE_EXPLORER_API', True)
        self.max_pages_per_address = getattr(settings, 'XONE_MAX_PAGES_PER_ADDRESS', 10)

        # Auto-configure Xone network settings
        self._configure_xone_settings()

        logger.info(f"XoneInternalTxIndexer initialized (chain_id={XONE_CHAIN_ID}, explorer_api={self.use_explorer_api})")
        logger.info("Xone L1 network support enabled automatically")

    def _configure_xone_settings(self):
        """Configure Xone network specific settings"""
        # Xone Safe contract addresses
        if not hasattr(settings, 'XONE_SAFE_CONTRACTS'):
            settings.XONE_SAFE_CONTRACTS = {
                'SAFE_SINGLETON': '0x883B88D417d289aF36782695364cCEe116fD1045',
                'SAFE_PROXY_FACTORY': '0x84767EAB2200A16606a6743c917803edb0485974',
                'MULTI_SEND': '0xa3dEea802b8D9bDF3c008ea8144737D40315a6A2',
                'MULTI_SEND_CALL_ONLY': '0x6cb727C047E9bc8A3Fb219A3704811950ce22C4B',
                'FALLBACK_HANDLER': '0x185ABEfe19Aa56eFAccd6819fE4605D804980D20',
            }
            logger.info(f"Configured Xone Safe contracts - Singleton: {settings.XONE_SAFE_CONTRACTS['SAFE_SINGLETON']}")

        # Adjust indexer settings for Xone (no trace API)
        if not hasattr(settings, 'XONE_CONFIGURED'):
            settings.ETH_INTERNAL_NO_FILTER = True
            settings.ETH_INTERNAL_TRACE_TXS_BATCH_SIZE = 0
            settings.XONE_CONFIGURED = True
            logger.debug("Disabled trace API features for Xone network")

    def _load_safe_addresses(self):
        """Load all Safe addresses from database"""
        self.safe_addresses = set(
            addr.lower() for addr in
            SafeContract.objects.all().values_list("address", flat=True)
        )
        logger.debug(f"Loaded {len(self.safe_addresses)} Safe addresses")

    def find_relevant_elements(
        self,
        addresses: set[ChecksumAddress],
        from_block_number: int,
        to_block_number: int,
        current_block_number: int | None = None,
    ) -> OrderedDict[bytes, None]:
        """
        Xone network doesn't support trace API, return empty dict.
        Actual indexing is done in process_elements using Explorer API.
        """
        logger.debug(f"XoneInternalTxIndexer.find_relevant_elements called for blocks {from_block_number}-{to_block_number}")

        # For Xone, we fetch Safe addresses directly from ProxyFactory via Explorer API
        self._fetch_safe_addresses_from_explorer()

        # Return empty OrderedDict, actual indexing happens in process_elements
        return OrderedDict()

    def _fetch_safe_addresses_from_explorer(self):
        """
        Fetch all Safe addresses from ProxyFactory logs via Explorer API.
        Uses Explorer API instead of eth_getLogs for better performance.
        Note: eth_getLogs works but is slow, Explorer API is much faster.
        Handles pagination to ensure all Safe addresses are retrieved.
        Only queries V1_4_1 ProxyFactory address (current deployment version).
        """
        # Get ProxyFactory address from environment variables
        # Only use V1_4_1 as it's the current deployment version
        proxy_factory_v1_4_1 = os.environ.get('SAFE_PROXY_FACTORY_V1_4_1', '0x84767EAB2200A16606a6743c917803edb0485974')

        proxy_factories = [
            ('V1_4_1', proxy_factory_v1_4_1),
        ]

        logger.info(f"Fetching Safe addresses from ProxyFactory V1_4_1: {proxy_factory_v1_4_1}")

        total_all_safes = 0

        for factory_version, proxy_factory in proxy_factories:
            try:
                endpoint = f"/addresses/{proxy_factory}/logs"
                page = 1
                next_page_params = None
                total_new_safes = 0

                logger.info(f"Querying ProxyFactory {factory_version} at {proxy_factory}")

                # Process all pages (with reasonable limit)
                while page <= 50:  # Safety limit to prevent infinite loops
                    response = _try_explorer_apis(endpoint, params=next_page_params)
                    if not response:
                        break

                    data = response.json()
                    items = data.get("items", [])

                    if not items:
                        break  # No more data

                    logger.debug(f"Processing ProxyFactory {factory_version} logs page {page}: {len(items)} entries")

                    new_safes_count = 0
                    safe_contracts_to_create = []

                    for item in items:
                        decoded = item.get("decoded", {})
                        method_call = decoded.get("method_call", "")
                        tx_hash = item.get("transaction_hash")

                        # Check if this is a ProxyCreation event
                        if "ProxyCreation" in method_call and tx_hash:
                            params = decoded.get("parameters", [])
                            for param in params:
                                if param.get("name") == "proxy":
                                    safe_address = param.get("value")
                                    if safe_address:
                                        safe_address_lower = safe_address.lower()
                                        if safe_address_lower not in self.safe_addresses:
                                            self.safe_addresses.add(safe_address_lower)

                                            # Prepare SafeContract for creation
                                            try:
                                                tx_hash_bytes = bytes.fromhex(tx_hash[2:] if tx_hash.startswith("0x") else tx_hash)
                                                safe_address_bytes = bytes.fromhex(safe_address[2:] if safe_address.startswith("0x") else safe_address)

                                                # Get or create EthereumTx first
                                                block_number = item.get("block_number", 0)

                                                # Get from_address, to_address, and input data by querying the transaction via RPC
                                                from_addr = b"\x00" * 20  # default
                                                to_addr = b""  # default
                                                tx_input_data = b""  # default
                                                try:
                                                    tx_data = self.ethereum_client.w3.eth.get_transaction(tx_hash)
                                                    if tx_data and tx_data.get('from') and isinstance(tx_data['from'], str):
                                                        from_addr = bytes.fromhex(tx_data['from'][2:] if tx_data['from'].startswith('0x') else tx_data['from'])
                                                    if tx_data and tx_data.get('to') and isinstance(tx_data['to'], str):
                                                        to_addr = bytes.fromhex(tx_data['to'][2:] if tx_data['to'].startswith('0x') else tx_data['to'])
                                                    if tx_data and tx_data.get('input') and isinstance(tx_data['input'], str):
                                                        input_hex = tx_data['input'][2:] if tx_data['input'].startswith('0x') else tx_data['input']
                                                        tx_input_data = bytes.fromhex(input_hex) if input_hex else b""
                                                    logger.debug(f"Retrieved from_addr from RPC: {tx_data.get('from') if tx_data else 'N/A'}")
                                                except Exception as e:
                                                    logger.warning(f"Failed to get transaction data from RPC for {tx_hash}: {e}")

                                                # Ensure EthereumBlock exists
                                                from ..models import EthereumBlock
                                                # Use block_number as placeholder hash (32 bytes)
                                                placeholder_hash = block_number.to_bytes(32, byteorder='big')
                                                # Use block_number - 1 as placeholder parent_hash
                                                placeholder_parent = (block_number - 1).to_bytes(32, byteorder='big') if block_number > 0 else b"\x00" * 32
                                                EthereumBlock.objects.get_or_create(
                                                    number=block_number,
                                                    defaults={
                                                        "gas_limit": 0,
                                                        "gas_used": 0,
                                                        "timestamp": datetime.now(timezone.utc),
                                                        "block_hash": placeholder_hash,
                                                        "parent_hash": placeholder_parent,
                                                    }
                                                )

                                                # Create or get EthereumTx
                                                ethereum_tx, _ = EthereumTx.objects.get_or_create(
                                                    tx_hash=tx_hash_bytes,
                                                    defaults={
                                                        "_from": from_addr,
                                                        "to": to_addr,
                                                        "value": 0,
                                                        "gas": 0,
                                                        "gas_price": 0,
                                                        "gas_used": 0,
                                                        "data": tx_input_data,
                                                        "nonce": 0,
                                                        "block_id": block_number,
                                                        "status": 1,
                                                    }
                                                )

                                                # Create SafeContract
                                                safe_contract, created = SafeContract.objects.get_or_create(
                                                    address=safe_address_bytes,
                                                    defaults={
                                                        "ethereum_tx": ethereum_tx,
                                                        "banned": False,
                                                    }
                                                )

                                                logger.info(f"SafeContract.get_or_create - address: {safe_address}, created: {created}")

                                                # Auto-create SafeLastStatus for new Safes
                                                if created:
                                                    logger.info(f"Attempting to create SafeLastStatus for new Safe: {safe_address}")
                                                    try:
                                                        from ..services import SafeServiceProvider
                                                        from ..models import SafeLastStatus, InternalTx, InternalTxType

                                                        logger.debug(f"Importing SafeServiceProvider for {safe_address}")
                                                        safe_service = SafeServiceProvider()

                                                        logger.debug(f"Fetching Safe info from blockchain for {safe_address}")
                                                        safe_info = safe_service.get_safe_info_from_blockchain(safe_address)

                                                        logger.debug(f"Safe info retrieved: nonce={safe_info.nonce}, threshold={safe_info.threshold}, owners={safe_info.owners}")

                                                        # Create InternalTx record for the Safe creation
                                                        # This is required for SafeLastStatus as it has a OneToOne relationship with InternalTx
                                                        internal_tx, internal_created = InternalTx.objects.get_or_create(
                                                            ethereum_tx=ethereum_tx,
                                                            trace_address="0",  # Root trace address for L2 synthetic traces
                                                            defaults={
                                                                'timestamp': ethereum_tx.block.timestamp,
                                                                'block_number': block_number,
                                                                '_from': from_addr,
                                                                'gas': 0,
                                                                'data': None,
                                                                'to': None,
                                                                'value': 0,
                                                                'gas_used': 0,
                                                                'contract_address': safe_address_bytes,
                                                                'code': None,
                                                                'output': None,
                                                                'refund_address': None,
                                                                'tx_type': InternalTxType.CREATE.value,
                                                                'call_type': None,
                                                                'error': None,
                                                            }
                                                        )
                                                        logger.debug(f"InternalTx created/retrieved: {internal_tx} (created={internal_created})")

                                                        status, status_created = SafeLastStatus.objects.get_or_create(
                                                            address=safe_address_bytes,
                                                            defaults={
                                                                'internal_tx': internal_tx,
                                                                'nonce': safe_info.nonce,
                                                                'threshold': safe_info.threshold,
                                                                'owners': safe_info.owners,
                                                                'master_copy': safe_info.master_copy,
                                                                'fallback_handler': safe_info.fallback_handler,
                                                                'guard': safe_info.guard,
                                                                'enabled_modules': [],
                                                            }
                                                        )
                                                        logger.info(f"✅ SafeLastStatus created/retrieved for {safe_address} (created={status_created})")
                                                    except Exception as status_err:
                                                        import traceback
                                                        logger.error(f"❌ Failed to create SafeLastStatus for {safe_address}: {status_err}")
                                                        logger.error(f"Traceback: {traceback.format_exc()}")
                                                else:
                                                    logger.debug(f"SafeContract already exists, skipping SafeLastStatus creation for {safe_address}")

                                                new_safes_count += 1
                                                logger.info(f"Created SafeContract record for: {safe_address}")

                                            except Exception as e:
                                                logger.error(f"Failed to create SafeContract for {safe_address}: {e}")
                                        else:
                                            logger.debug(f"Safe address already indexed: {safe_address}")

                    if new_safes_count > 0:
                        logger.info(f"Found {new_safes_count} new Safe addresses from ProxyFactory {factory_version} (page {page})")
                        total_new_safes += new_safes_count

                    # Check for next page
                    next_page_params = data.get("next_page_params")
                    if not next_page_params:
                        break  # No more pages

                    page += 1

                if total_new_safes > 0:
                    logger.info(f"ProxyFactory {factory_version}: Found {total_new_safes} new Safe addresses across {page} pages")
                    total_all_safes += total_new_safes

            except Exception as e:
                logger.error(f"Failed to fetch Safe addresses from ProxyFactory {factory_version}: {e}")

        if total_all_safes > 0:
            logger.info(f"Total Safe addresses found from all ProxyFactory contracts: {total_all_safes}")

    def _index_from_explorer_api(self, safe_address: str) -> int:
        """Index internal transactions and native transfers using Explorer API with pagination support"""
        if not self.use_explorer_api:
            return 0

        indexed_count = 0

        # Index internal transactions with pagination
        try:
            page = 1
            next_page_params = None

            while page <= self.max_pages_per_address:
                # Build endpoint with pagination
                endpoint = f"/addresses/{safe_address}/internal-transactions"
                response = _try_explorer_apis(endpoint, params=next_page_params)

                if not response:
                    break

                data = response.json()
                items = data.get("items", [])

                if not items:
                    break  # No more data

                logger.debug(f"Processing page {page} for {safe_address}: {len(items)} internal txs")

                for item in items:
                    # Only process call type with value
                    if item.get("type") != "call" or int(item.get("value", 0)) == 0:
                        continue

                    tx_hash = bytes.fromhex(item.get("transaction_hash", "0x")[2:])
                    if not tx_hash:
                        continue

                    # Check if already exists
                    if InternalTx.objects.filter(ethereum_tx_id=tx_hash).exists():
                        continue

                    # Create records
                    with transaction.atomic():
                        block_number = item.get("block_number", 0)
                        from_addr = bytes.fromhex(item["from"]["hash"][2:])
                        to_addr = bytes.fromhex(item.get("to", {}).get("hash", "0x")[2:]) if item.get("to") else None

                        # Ensure EthereumBlock exists first
                        # Use block_number as placeholder hash (32 bytes)
                        placeholder_hash = block_number.to_bytes(32, byteorder='big')
                        # Use block_number - 1 as placeholder parent_hash
                        placeholder_parent = (block_number - 1).to_bytes(32, byteorder='big') if block_number > 0 else b"\x00" * 32
                        EthereumBlock.objects.get_or_create(
                            number=block_number,
                            defaults={
                                "gas_limit": 0,
                                "gas_used": 0,
                                "timestamp": datetime.now(timezone.utc),
                                "block_hash": placeholder_hash,
                                "parent_hash": placeholder_parent,
                            }
                        )

                        # Ensure EthereumTx exists
                        ethereum_tx, _ = EthereumTx.objects.get_or_create(
                            tx_hash=tx_hash,
                            defaults={
                                "_from": from_addr,
                                "to": to_addr,
                                "value": 0,
                                "gas": 0,
                                "gas_price": 0,
                                "gas_used": 0,
                                "data": b"",
                                "nonce": 0,
                                "block_id": block_number,
                                "status": 1,
                            }
                        )

                        # Create InternalTx with timestamp
                        # Parse timestamp from Explorer API or use current time
                        timestamp_str = item.get("timestamp")
                        if timestamp_str:
                            from dateutil import parser as dateutil_parser
                            timestamp = dateutil_parser.parse(timestamp_str)
                        else:
                            timestamp = datetime.now(timezone.utc)

                        InternalTx.objects.create(
                            ethereum_tx=ethereum_tx,
                            _from=from_addr,
                            to=to_addr,
                            value=int(item.get("value", 0)),
                            gas=0,
                            gas_used=0,
                            data=b"",
                            output=b"",
                            trace_address="0",
                            tx_type=0,
                            call_type=0,
                            block_number=block_number,
                            timestamp=timestamp,
                        )

                        # Create SafeRelevantTransaction with timestamp
                        SafeRelevantTransaction.objects.get_or_create(
                            ethereum_tx=ethereum_tx,
                            safe=bytes.fromhex(safe_address[2:]),
                            defaults={
                                "timestamp": timestamp
                            }
                        )

                        indexed_count += 1

                # Check for next page
                next_page_params = data.get("next_page_params")
                if not next_page_params:
                    break  # No more pages

                page += 1

        except Exception as e:
            logger.error(f"Failed to index internal txs from explorer API for {safe_address}: {e}")

        # Also index normal transactions with pagination
        try:
            page = 1
            next_page_params = None

            while page <= self.max_pages_per_address:
                endpoint = f"/addresses/{safe_address}/transactions"
                response = _try_explorer_apis(endpoint, params=next_page_params)

                if not response:
                    break

                data = response.json()
                items = data.get("items", [])

                if not items:
                    break

                logger.debug(f"Processing page {page} for {safe_address}: {len(items)} transactions")

                for item in items:
                    # Only process transactions with value
                    if int(item.get("value", 0)) == 0:
                        continue

                    tx_hash = bytes.fromhex(item.get("hash", "0x")[2:])
                    if not tx_hash:
                        continue

                    # Check if already exists
                    if InternalTx.objects.filter(ethereum_tx_id=tx_hash).exists():
                        continue

                    # Create records for native transfers
                    with transaction.atomic():
                        block_number = item.get("block_number", 0)
                        from_addr = bytes.fromhex(item["from"]["hash"][2:])
                        to_addr = bytes.fromhex(item.get("to", {}).get("hash", "0x")[2:]) if item.get("to") else None

                        # Ensure EthereumBlock exists first
                        # Use block_number as placeholder hash (32 bytes)
                        placeholder_hash = block_number.to_bytes(32, byteorder='big')
                        # Use block_number - 1 as placeholder parent_hash
                        placeholder_parent = (block_number - 1).to_bytes(32, byteorder='big') if block_number > 0 else b"\x00" * 32
                        EthereumBlock.objects.get_or_create(
                            number=block_number,
                            defaults={
                                "gas_limit": 0,
                                "gas_used": 0,
                                "timestamp": datetime.now(timezone.utc),
                                "block_hash": placeholder_hash,
                                "parent_hash": placeholder_parent,
                            }
                        )

                        # Ensure EthereumTx exists
                        ethereum_tx, _ = EthereumTx.objects.get_or_create(
                            tx_hash=tx_hash,
                            defaults={
                                "_from": from_addr,
                                "to": to_addr,
                                "value": int(item.get("value", 0)),
                                "gas": int(item.get("gas_limit", 0)),
                                "gas_price": int(item.get("gas_price", 0)),
                                "gas_used": int(item.get("gas_used", 0)),
                                "data": bytes.fromhex(item.get("raw_input", "0x")[2:]),
                                "nonce": item.get("nonce", 0),
                                "block_id": block_number,
                                "status": 1 if item.get("status") == "ok" else 0,
                            }
                        )

                        # Create InternalTx for native transfer with timestamp
                        # Parse timestamp from Explorer API or use current time
                        timestamp_str = item.get("timestamp")
                        if timestamp_str:
                            from dateutil import parser as dateutil_parser
                            timestamp = dateutil_parser.parse(timestamp_str)
                        else:
                            timestamp = datetime.now(timezone.utc)

                        InternalTx.objects.create(
                            ethereum_tx=ethereum_tx,
                            _from=from_addr,
                            to=to_addr,
                            value=int(item.get("value", 0)),
                            gas=int(item.get("gas_limit", 0)),
                            gas_used=int(item.get("gas_used", 0)),
                            data=b"",
                            output=b"",
                            trace_address="0",
                            tx_type=0,
                            call_type=0,
                            block_number=block_number,
                            timestamp=timestamp,
                        )

                        # Create SafeRelevantTransaction with timestamp
                        SafeRelevantTransaction.objects.get_or_create(
                            ethereum_tx=ethereum_tx,
                            safe=bytes.fromhex(safe_address[2:]),
                            defaults={
                                "timestamp": timestamp
                            }
                        )

                        indexed_count += 1

                # Check for next page
                next_page_params = data.get("next_page_params")
                if not next_page_params:
                    break

                page += 1

        except Exception as e:
            logger.error(f"Failed to index transactions from explorer API for {safe_address}: {e}")

        if indexed_count > 0:
            logger.info(f"Indexed {indexed_count} total transactions from explorer API for {safe_address}")

        return indexed_count

    def process_elements(
        self, tx_hash_with_traces: OrderedDict[bytes, FilterTrace | None]
    ) -> list[HexBytes]:
        """
        Override process_elements to use Explorer API for data fetching.
        Falls back to parent implementation if trace data is available.
        """
        # If we have trace data, use parent's processing method
        if tx_hash_with_traces:
            return super().process_elements(tx_hash_with_traces)

        # Otherwise use Explorer API
        if not self.safe_addresses:
            self._load_safe_addresses()

        logger.info(f"Processing {len(self.safe_addresses)} Safes using Explorer API")

        total_indexed = 0
        for safe_addr in self.safe_addresses:
            safe_hex = safe_addr if safe_addr.startswith("0x") else f"0x{safe_addr}"
            indexed = self._index_from_explorer_api(safe_hex)
            total_indexed += indexed

        if total_indexed > 0:
            logger.info(f"Total indexed from Explorer API: {total_indexed} transactions")

        return []

    def update_monitored_addresses(
        self, addresses: set[str], from_block_number: int, to_block_number: int
    ) -> bool:
        """
        Override for XoneInternalTxIndexer - more lenient with newly discovered Safes.
        For Xone Network, new Safes can be discovered during indexing through ProxyFactory
        events. These new Safes may have been initialized at any block number, not
        necessarily within the current indexing range. This is expected behavior and
        should not be treated as a reorg.
        """
        logger.debug(
            "%s: Updating monitored addresses (Xone mode - lenient for new Safes)",
            self.__class__.__name__,
        )

        # Keep indexing going on the next block
        new_to_block_number = to_block_number + 1

        updated_addresses = self.database_queryset.filter(
            **{
                "address__in": addresses,
                self.database_field + "__gte": from_block_number,
                self.database_field + "__lt": new_to_block_number,
            }
        ).update(**{self.database_field: new_to_block_number})

        # For Xone, we're lenient - if we updated at least one address, that's fine
        # New Safes discovered during indexing may not be in the range
        if updated_addresses < len(addresses):
            logger.info(
                "%s: Updated %d/%d addresses. This is normal for Xone - "
                "new Safes may have been discovered with different block numbers. "
                "from-block-number=%d to-block-number=%d",
                self.__class__.__name__,
                updated_addresses,
                len(addresses),
                from_block_number,
                new_to_block_number,
            )

        logger.debug(
            "%s: Updated monitored addresses",
            self.__class__.__name__,
        )

        # Always return True for Xone - discovering new Safes is expected behavior
        return True
