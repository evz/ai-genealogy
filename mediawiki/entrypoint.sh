#!/bin/bash
set -e

SETTINGS_FILE="/var/www/html/LocalSettings.php"
SETTINGS_PERSIST="/wiki-config/LocalSettings.php"

# On subsequent starts, restore LocalSettings.php from persistent volume
if [ ! -f "$SETTINGS_FILE" ] && [ -f "$SETTINGS_PERSIST" ]; then
    cp "$SETTINGS_PERSIST" "$SETTINGS_FILE"
fi

if [ ! -f "$SETTINGS_FILE" ]; then
    echo "=== First run: installing MediaWiki ==="
    php /var/www/html/maintenance/run.php install \
        --dbserver "${WIKI_DB_HOST}" \
        --dbname "${WIKI_DB_NAME}" \
        --dbuser "${WIKI_DB_USER}" \
        --dbpass "${WIKI_DB_PASSWORD}" \
        --pass "${WIKI_ADMIN_PASS}" \
        --scriptpath "" \
        --lang "en" \
        "${WIKI_SITE_NAME}" \
        "${WIKI_ADMIN_USER}"

    cat >> "$SETTINGS_FILE" << 'SETTINGS'

# Pretty URLs
$wgArticlePath = '/wiki/$1';
$wgUsePathInfo = true;

# Enable API writes
$wgEnableWriteAPI = true;

# Allow longer article titles for genealogical IDs
$wgMaxArticleSize = 4096;
SETTINGS

    # Persist LocalSettings.php so it survives container restarts
    cp "$SETTINGS_FILE" "$SETTINGS_PERSIST"
    echo "=== Installation complete. Wiki available at http://localhost:8081 ==="
fi

exec apache2-foreground
