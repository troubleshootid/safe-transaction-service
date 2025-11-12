#!/bin/bash
set -e

# Wait for database
echo "Waiting for database..."
while ! nc -z ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432}; do
  sleep 1
done

# Try to run migrations, but continue even if they fail
echo "Running migrations..."
python manage.py migrate --noinput || {
    echo "Migrations failed, attempting to reset..."
    python manage.py migrate --fake-initial --noinput || {
        echo "Migration reset failed, continuing anyway..."
    }
}

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput || echo "Static collection failed, continuing..."

# Create superuser if needed (optional)
echo "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.filter(username='admin').exists() or User.objects.create_superuser('admin', 'admin@example.com', 'admin123')" | python manage.py shell || echo "Superuser creation skipped"

# Initialize Xone Network contracts configuration
echo "Initializing Xone Network contracts..."
python manage.py shell <<EOF || echo "Xone Network configuration skipped"
from safe_transaction_service.history.models import SafeMasterCopy, ProxyFactory
import os

# Only configure if chain ID is 3721 (Xone Network)
if os.environ.get('ETHEREUM_CHAIN_ID') == '3721':
    print('Configuring Xone Network (Chain ID: 3721)...')

    # Safe Master Copy 1.4.1
    SafeMasterCopy.objects.get_or_create(
        address='0x883B88D417d289aF36782695364cCEe116fD1045',
        defaults={
            'initial_block_number': 15000000,
            'tx_block_number': 15000000,
            'version': '1.4.1',
            'l2': False,
            'deployer': '0x0000000000000000000000000000000000000000'
        }
    )
    print('SafeMasterCopy 1.4.1 configured')

    # Proxy Factory 1.4.1
    ProxyFactory.objects.get_or_create(
        address='0x84767EAB2200A16606a6743c917803edb0485974',
        defaults={
            'initial_block_number': 15000000,
            'tx_block_number': 15000000
        }
    )
    print('ProxyFactory 1.4.1 configured')

    print('Xone Network contracts configuration completed')
else:
    print(f"Chain ID is {os.environ.get('ETHEREUM_CHAIN_ID')}, not Xone Network")
EOF

# Setup periodic tasks for the indexer
echo "Setting up indexer periodic tasks..."
python manage.py setup_service || echo "Periodic tasks setup skipped"

# Start server
echo "Starting server..."
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:${GUNICORN_BIND_PORT:-8888} \
    --workers ${WEB_CONCURRENCY:-4} \
    --worker-class gunicorn_custom_workers.MyGeventWorker \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info