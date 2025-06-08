# TGMulticanal Bot

A Telegram bot that posts updates from Twitter and Instagram into your channels. It also includes basic security features and an optional AI chat module.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a Telegram bot and obtain its API token.
3. Set the environment variable `BOT_TOKEN` with your token.
4. Run the bot:
   ```bash
   python telegrambot.py
   ```

## Commands

- `/start` – show the main menu (admins only)
- `/members` – list member count for each channel
- `/post` – force immediate posting for a channel
- `/setwelcome <channel_id> on|off [text]` – configure welcome messages
- `/learn question | answer` – teach the bot a reply
- `/setsecurity <channel_id> on|off` – toggle auto-banning on a channel
- `/banned <channel_id>` – list banned users

Use the inline menus to manage channels, links, admins and advertisement messages.
