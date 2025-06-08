import os
import sqlite3
import logging
import re
import json
from datetime import datetime, timedelta

import aiohttp
from aiohttp import http_exceptions
from bs4 import BeautifulSoup
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)
import asyncio

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable not set")

# Nombre maximal de posts à récupérer pour chaque lien
POSTS_LIMIT = 10

SUPERADMIN_ID = 5310947336
DB_NAME = "botfull.db"

# ====== BASE & DB STRUCTURE ======

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS admins (
        user_id INTEGER PRIMARY KEY,
        username TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS channels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        chat_id TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS channels_admins (
        channel_id INTEGER,
        admin_id INTEGER,
        PRIMARY KEY(channel_id, admin_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id INTEGER,
        platform TEXT,
        url TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        link_id INTEGER,
        post_id TEXT,
        date_postee TEXT,
        deja_poste INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS stats (
        channel_id INTEGER PRIMARY KEY,
        last_count INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS welcome_channels (
        channel_id INTEGER PRIMARY KEY,
        welcome_text TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS ads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        interval_hours INTEGER DEFAULT 24,
        last_sent TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS learn_data (
        question TEXT,
        answer TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS security_settings (
        channel_id INTEGER PRIMARY KEY,
        auto_ban INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS banned_users (
        channel_id INTEGER,
        user_id INTEGER,
        username TEXT,
        reason TEXT,
        banned_on TEXT,
        PRIMARY KEY(channel_id, user_id)
    )''')
    conn.commit()
    conn.close()


def get_admins():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT user_id FROM admins')
    res = [x[0] for x in c.fetchall()]
    conn.close()
    return list(set(res + [SUPERADMIN_ID]))


def add_admin(user_id, username=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO admins (user_id, username) VALUES (?, ?)', (user_id, username))
    conn.commit()
    conn.close()


def delete_admin(user_id):
    if user_id == SUPERADMIN_ID:
        return
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM admins WHERE user_id=?', (user_id,))
    c.execute('DELETE FROM channels_admins WHERE admin_id=?', (user_id,))
    conn.commit()
    conn.close()


def get_channels(admin_id=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if admin_id and admin_id != SUPERADMIN_ID:
        c.execute('''SELECT ch.id, ch.name, ch.chat_id FROM channels ch
                     JOIN channels_admins ca ON ch.id = ca.channel_id
                     WHERE ca.admin_id=?''', (admin_id,))
    else:
        c.execute('SELECT id, name, chat_id FROM channels')
    rows = c.fetchall()
    conn.close()
    return rows


def add_channel(name, chat_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO channels (name, chat_id) VALUES (?, ?)', (name, chat_id))
    channel_id = c.lastrowid
    conn.commit()
    conn.close()
    return channel_id


def delete_channel(channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM channels WHERE id = ?', (channel_id,))
    c.execute('DELETE FROM channels_admins WHERE channel_id = ?', (channel_id,))
    c.execute('DELETE FROM links WHERE channel_id = ?', (channel_id,))
    conn.commit()
    conn.close()


def assign_admin_to_channel(admin_id, channel_id):
    """Assign an admin to a channel and ensure they are registered as an admin."""
    add_admin(admin_id)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        'INSERT OR IGNORE INTO channels_admins (channel_id, admin_id) VALUES (?, ?)',
        (channel_id, admin_id),
    )
    conn.commit()
    conn.close()


def remove_admin_from_channel(admin_id, channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM channels_admins WHERE channel_id=? AND admin_id=?', (channel_id, admin_id))
    conn.commit()
    conn.close()


def get_channel_admins(channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT admin_id FROM channels_admins WHERE channel_id=?', (channel_id,))
    res = [x[0] for x in c.fetchall()]
    conn.close()
    return res


def get_links(channel_id=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if channel_id:
        c.execute('SELECT id, platform, url FROM links WHERE channel_id = ?', (channel_id,))
    else:
        c.execute('SELECT id, platform, url, channel_id FROM links')
    rows = c.fetchall()
    conn.close()
    return rows


def add_link(channel_id, platform, url):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO links (channel_id, platform, url) VALUES (?, ?, ?)', (channel_id, platform, url))
    conn.commit()
    conn.close()


def update_link(link_id, new_url):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE links SET url = ? WHERE id = ?', (new_url, link_id))
    conn.commit()
    conn.close()


def delete_link(link_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM links WHERE id = ?', (link_id,))
    c.execute('DELETE FROM posts WHERE link_id = ?', (link_id,))
    conn.commit()
    conn.close()


def get_channel(channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id, name, chat_id FROM channels WHERE id = ?', (channel_id,))
    row = c.fetchone()
    conn.close()
    return row


def get_link(link_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id, platform, url, channel_id FROM links WHERE id = ?', (link_id,))
    row = c.fetchone()
    conn.close()
    return row


def post_exists(link_id, post_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id FROM posts WHERE link_id = ? AND post_id = ?', (link_id, post_id))
    res = c.fetchone()
    conn.close()
    return res is not None


def mark_posted(link_id, post_id):
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO posts (link_id, post_id, date_postee, deja_poste) VALUES (?, ?, ?, ?)', (link_id, post_id, now, 1))
    conn.commit()
    conn.close()


def add_old_posts(link_id, post_ids):
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    for post_id in post_ids:
        if not post_exists(link_id, post_id):
            c.execute('INSERT INTO posts (link_id, post_id, date_postee, deja_poste) VALUES (?, ?, ?, ?)', (link_id, post_id, now, 0))
    conn.commit()
    conn.close()


def get_unposted():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''SELECT posts.id, posts.link_id, posts.post_id, links.platform, links.url, links.channel_id
           FROM posts JOIN links ON posts.link_id = links.id
           WHERE posts.deja_poste = 0''')
    rows = c.fetchall()
    conn.close()
    return rows


def set_posted(post_db_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE posts SET deja_poste = 1 WHERE id = ?', (post_db_id,))
    conn.commit()
    conn.close()


def set_last_stats(channel_id, count):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO stats (channel_id, last_count) VALUES (?, ?)', (channel_id, count))
    conn.commit()
    conn.close()


def get_last_stats(channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT last_count FROM stats WHERE channel_id=?', (channel_id,))
    r = c.fetchone()
    conn.close()
    return r[0] if r else None


def set_welcome(channel_id, text):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO welcome_channels (channel_id, welcome_text) VALUES (?, ?)', (channel_id, text))
    conn.commit()
    conn.close()


def disable_welcome(channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM welcome_channels WHERE channel_id=?', (channel_id,))
    conn.commit()
    conn.close()


def get_welcome_text(channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT welcome_text FROM welcome_channels WHERE channel_id=?', (channel_id,))
    r = c.fetchone()
    conn.close()
    return r[0] if r else None


def set_auto_ban(channel_id, state: bool):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        'INSERT OR REPLACE INTO security_settings (channel_id, auto_ban) VALUES (?, ?)',
        (channel_id, int(state)),
    )
    conn.commit()
    conn.close()


def get_auto_ban(channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT auto_ban FROM security_settings WHERE channel_id=?', (channel_id,))
    r = c.fetchone()
    conn.close()
    return bool(r[0]) if r else False


def ban_user_record(channel_id, user_id, username, reason):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        'INSERT OR REPLACE INTO banned_users (channel_id, user_id, username, reason, banned_on) VALUES (?, ?, ?, ?, ?)',
        (channel_id, user_id, username, reason, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_banned(channel_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT user_id, username FROM banned_users WHERE channel_id=?', (channel_id,))
    rows = c.fetchall()
    conn.close()
    return rows


def is_user_banned(channel_id, user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT 1 FROM banned_users WHERE channel_id=? AND user_id=?', (channel_id, user_id))
    r = c.fetchone()
    conn.close()
    return r is not None


def add_ad(text, interval_hours=24):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO ads (text, interval_hours, last_sent) VALUES (?, ?, ?)', (text, interval_hours, None))
    conn.commit()
    conn.close()


def get_ads():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id, text, interval_hours, last_sent FROM ads')
    rows = c.fetchall()
    conn.close()
    return rows


def delete_ad(ad_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM ads WHERE id=?', (ad_id,))
    conn.commit()
    conn.close()


def update_ad(ad_id, text=None, interval_hours=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if text is not None:
        c.execute('UPDATE ads SET text=? WHERE id=?', (text, ad_id))
    if interval_hours is not None:
        c.execute('UPDATE ads SET interval_hours=? WHERE id=?', (interval_hours, ad_id))
    conn.commit()
    conn.close()


def due_ads():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now()
    rows = []
    for row in c.execute('SELECT id, text, interval_hours, last_sent FROM ads'):
        ad_id, text, interval_hours, last_sent = row
        if not last_sent:
            rows.append(row)
        else:
            try:
                last = datetime.fromisoformat(last_sent)
            except Exception:
                last = now
            if now - last >= timedelta(hours=interval_hours or 24):
                rows.append(row)
    conn.close()
    return rows


def mark_ad_sent(ad_id):
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE ads SET last_sent=? WHERE id=?', (now, ad_id))
    conn.commit()
    conn.close()


def add_learn(question, answer):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO learn_data (question, answer) VALUES (?, ?)', (question, answer))
    conn.commit()
    conn.close()


def get_learn_data():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT question, answer FROM learn_data')
    data = c.fetchall()
    conn.close()
    return data


def get_learn_answer(text):
    data = get_learn_data()
    if not data:
        return None
    questions = [d[0] for d in data]
    vectorizer = TfidfVectorizer().fit(questions)
    vectors = vectorizer.transform(questions + [text])
    sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    idx = sims.argmax()
    if sims[idx] > 0.5:
        return data[idx][1]
    return None


async def check_channel_access(context, chat_id):
    try:
        chat = await context.bot.get_chat(chat_id)
        member = await context.bot.get_chat_member(chat_id, context.bot.id)
        can_post = member.status in ("administrator", "creator")
        return (True, can_post, chat)
    except Exception as e:
        return (False, False, str(e))


async def fetch_html(url):
    """Fetch HTML using aiohttp with a requests fallback for large headers."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as resp:
                return await resp.text()
    except Exception as e:
        logger.warning("aiohttp fetch failed for %s: %s", url, e)
        try:
            return await asyncio.to_thread(lambda: requests.get(url, headers=headers, timeout=10).text)
        except Exception as e2:
            logger.error("requests fetch failed for %s: %s", url, e2)
            return ""


def extract_tweet_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    tweet_text_tag = soup.find(attrs={"data-testid": "tweetText"})
    tweet_text = tweet_text_tag.get_text(separator=' ').strip() if tweet_text_tag else ""
    mentions = []
    if tweet_text_tag:
        for a in tweet_text_tag.find_all("a"):
            href = a.get("href")
            if href and href.startswith("/"):
                username = href.split('/')[1]
                mentions.append(f"@{username}")
    img_tag = soup.find("img", {"alt": "Image"})
    img_url = img_tag.get("src") if img_tag else None
    user_tag = soup.find("a", href=True, attrs={"role": "link"})
    author = "@" + user_tag.get("href").strip('/') if user_tag else ""
    txt = f"{tweet_text}\n\n👤 {author}"
    if mentions:
        txt += "\n🔗 " + " ".join(set(mentions))
    return txt, img_url


def extract_instagram_from_html(html):
    """Parse an Instagram post page HTML to get caption, author and image."""
    soup = BeautifulSoup(html, 'html.parser')
    caption = ""
    img_url = None
    author = ""
    # Try JSON-LD data first as it contains structured info
    ld_json = soup.find("script", type="application/ld+json")
    if ld_json and ld_json.string:
        try:
            data = json.loads(ld_json.string)
            caption = data.get("caption", "")
            img_url = data.get("image")
            author_name = data.get("author", {}).get("alternateName")
            if author_name:
                author = f"@{author_name.lstrip('@')}"
        except Exception:
            pass
    # Fallback to meta tags if JSON-LD is missing
    if not caption:
        meta_desc = soup.find("meta", property="og:description")
        if meta_desc:
            m = re.search(r'Instagram: "([^"]+)"', meta_desc.get("content", ""))
            if m:
                caption = m.group(1)
    if not img_url:
        img_meta = soup.find("meta", property="og:image")
        if img_meta:
            img_url = img_meta.get("content")
    mentions = re.findall(r'@([A-Za-z0-9_.]+)', caption)
    text = caption.strip()
    if author:
        text += f"\n\n👤 {author}"
    if mentions:
        text += "\n🔗 " + " ".join(f"@{m}" for m in set(mentions))
    return text, img_url


async def scrape_twitter(url, max_posts=POSTS_LIMIT):
    try:
        html = await fetch_html(url)
        found = re.findall(r'/status/(\d+)', html)
        unique = []
        [unique.append(x) for x in found if x not in unique]
        return unique[:max_posts]
    except Exception as e:
        logger.error(f"scrape_twitter failed for {url}: {e}")
        return []


async def scrape_instagram(url, max_posts=POSTS_LIMIT):
    try:
        html = await fetch_html(url)
        found = re.findall(r'"shortcode":"([^"]+)"', html)
        unique = []
        [unique.append(x) for x in found if x not in unique]
        return unique[:max_posts]
    except Exception as e:
        logger.error(f"scrape_instagram failed for {url}: {e}")
        return []


def get_post_url(platform, user_url, post_id):
    if platform == "Twitter":
        username = user_url.rstrip('/').split('/')[-1]
        return f"https://x.com/{username}/status/{post_id}"
    if platform == "Instagram":
        return f"https://www.instagram.com/p/{post_id}/"
    return ""


async def send_post_details(context, chat_id, platform, post_url):
    try:
        html = await fetch_html(post_url)
        if platform == "Twitter":
            text, img = extract_tweet_from_html(html)
        elif platform == "Instagram":
            text, img = extract_instagram_from_html(html)
        else:
            text, img = "", None
        message = text.strip() if text else ""
        if message:
            message += f"\n\n🔗 {post_url}"
            await context.bot.send_message(chat_id=chat_id, text=message)
        else:
            await context.bot.send_message(chat_id=chat_id, text=post_url)
        if img:
            await context.bot.send_photo(chat_id=chat_id, photo=img)
    except Exception as e:
        logger.error(f"send_post_details failed for {post_url}: {e}")
        await context.bot.send_message(chat_id=chat_id, text=post_url)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    AI_TOKENIZER = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    AI_MODEL = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    AI_HISTORY = {}
except Exception as e:
    logger.error("AI init failed: %s", e)
    AI_TOKENIZER = None
    AI_MODEL = None
    AI_HISTORY = {}


def ai_reply(user_id: int, text: str) -> str:
    if not AI_MODEL or not AI_TOKENIZER:
        return "Module IA indisponible."
    new_ids = AI_TOKENIZER.encode(text + AI_TOKENIZER.eos_token, return_tensors="pt")
    history = AI_HISTORY.get(user_id)
    if history is not None:
        bot_input_ids = torch.cat([history, new_ids], dim=-1)
    else:
        bot_input_ids = new_ids
    output_ids = AI_MODEL.generate(bot_input_ids, max_length=200, pad_token_id=AI_TOKENIZER.eos_token_id)
    AI_HISTORY[user_id] = output_ids
    reply = AI_TOKENIZER.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply.strip()


def is_authorized(user_id, channel_id=None):
    if channel_id:
        admins = get_channel_admins(channel_id)
        return user_id in admins or user_id == SUPERADMIN_ID
    return user_id in get_admins()


### ========== UI & MENUS ==========

async def show_main_menu(target):
    keyboard = [
        [InlineKeyboardButton("📋 Gérer les canaux", callback_data="menu_channels")],
        [InlineKeyboardButton("🔗 Gérer les liens", callback_data="menu_links")],
        [InlineKeyboardButton("🕒 Ajouter anciens posts", callback_data="add_oldposts")],
        [InlineKeyboardButton("🛡 Gérer les admins", callback_data="menu_admins")],
        [InlineKeyboardButton("📎 Assignation admins/canaux", callback_data="assign_menu")],
        [InlineKeyboardButton("📢 Gérer pubs", callback_data="menu_ads")],
        [InlineKeyboardButton("🔍 Tester envoi", callback_data="test_posting")]
    ]
    markup = InlineKeyboardMarkup(keyboard)
    if hasattr(target, "message"):
        await target.message.reply_text("Menu principal :", reply_markup=markup)
    elif hasattr(target, "edit_message_text"):
        await target.edit_message_text("Menu principal :", reply_markup=markup)
    elif hasattr(target, "reply_text"):
        await target.reply_text("Menu principal :", reply_markup=markup)


async def always_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user and is_authorized(update.effective_user.id):
        await show_main_menu(update)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔️ Accès refusé.")
        return
    await show_main_menu(update)


async def cmd_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⛔️ Accès refusé.")
        return
    channels = get_channels(admin_id=user_id if user_id != SUPERADMIN_ID else None)
    lines = []
    for chan in channels:
        try:
            count = await context.bot.get_chat_members_count(chan[2])
        except Exception as e:
            logger.error(f"Erreur get_chat_members_count {chan[2]}: {e}")
            count = "?"
        lines.append(f"{chan[1]} ({chan[2]}) : {count}")
    await update.message.reply_text("\n".join(lines) if lines else "Aucun canal")


async def cmd_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /post <channel_id> [nombre]")
        return
    try:
        channel_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Channel id invalide")
        return
    limit = int(context.args[1]) if len(context.args) > 1 else POSTS_LIMIT
    if not is_authorized(user_id, channel_id):
        await update.message.reply_text("⛔️ Accès refusé.")
        return
    await post_all_now(channel_id, context, limit)
    await update.message.reply_text("✅ Postage terminé")


async def cmd_setwelcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /setwelcome <channel_id> on|off [message]")
        return
    try:
        channel_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Channel id invalide")
        return
    if not is_authorized(user_id, channel_id):
        await update.message.reply_text("⛔️ Accès refusé.")
        return
    action = context.args[1].lower()
    if action == "on":
        text = " ".join(context.args[2:]) or "Bienvenue aux nouveaux membres !"
        set_welcome(channel_id, text)
        await update.message.reply_text("Welcome activé")
    elif action == "off":
        disable_welcome(channel_id)
        await update.message.reply_text("Welcome désactivé")
    else:
        await update.message.reply_text("Paramètre invalide: on ou off")


async def cmd_learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("⛔️ Accès refusé.")
        return
    if not context.args or "|" not in " ".join(context.args):
        await update.message.reply_text("Usage: /learn question | réponse")
        return
    raw = " ".join(context.args)
    question, answer = map(str.strip, raw.split("|", 1))
    add_learn(question, answer)
    await update.message.reply_text("✅ Appris !")


async def cmd_setsecurity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /setsecurity <channel_id> on|off")
        return
    try:
        channel_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Channel id invalide")
        return
    if not is_authorized(user_id, channel_id):
        await update.message.reply_text("⛔️ Accès refusé.")
        return
    state = context.args[1].lower() == "on"
    set_auto_ban(channel_id, state)
    await update.message.reply_text(
        f"Sécurité {'activée' if state else 'désactivée'} pour {channel_id}"
    )


async def cmd_banned(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /banned <channel_id>")
        return
    try:
        channel_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Channel id invalide")
        return
    if not is_authorized(user_id, channel_id):
        await update.message.reply_text("⛔️ Accès refusé.")
        return
    banned = get_banned(channel_id)
    if banned:
        txt = "\n".join(f"{uid} {uname or ''}" for uid, uname in banned)
    else:
        txt = "Aucun bannissement"
    await update.message.reply_text(txt)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if not is_authorized(user_id):
        await query.edit_message_text("⛔️ Accès refusé.")
        return

    if data == "menu_channels":
        await menu_channels(query, context, user_id)
    elif data.startswith("add_channel"):
        context.user_data["add_channel"] = True
        await query.edit_message_text("Nom + @chat_id OU -100... OU https://t.me/... (Ex: Glamboss https://t.me/+abcdefgHIKLMN)")
    elif data.startswith("delete_channel|"):
        channel_id = int(data.split("|")[1])
        delete_channel(channel_id)
        await menu_channels(query, context, user_id)
    elif data == "menu_links":
        await menu_links_select_channel(query, user_id=user_id)
    elif data.startswith("links_channel|"):
        channel_id = int(data.split("|")[1])
        await menu_links(query, channel_id, user_id)
    elif data.startswith("add_link|"):
        channel_id = int(data.split("|")[1])
        context.user_data["add_link"] = channel_id
        await query.edit_message_text("Plateforme (Twitter/Instagram) + URL du profil\nEx: Twitter https://x.com/xxx")
    elif data.startswith("delete_link|"):
        link_id, channel_id = map(int, data.split("|")[1:])
        delete_link(link_id)
        await menu_links(query, channel_id, user_id)
    elif data.startswith("edit_link|"):
        link_id, channel_id = map(int, data.split("|")[1:])
        context.user_data["edit_link"] = (link_id, channel_id)
        await query.edit_message_text("Envoie le nouveau lien complet (ex: Twitter https://x.com/xxx)")
    elif data == "add_oldposts":
        await menu_links_select_channel(query, for_oldposts=True, user_id=user_id)
    elif data.startswith("oldposts_links_channel|"):
        channel_id = int(data.split("|")[1])
        links = get_links(channel_id)
        if not links:
            await query.edit_message_text("Aucun lien sur ce canal.")
            return
        kb = []
        for lnk in links:
            kb.append([InlineKeyboardButton(f"{lnk[1]} - {lnk[2]}", callback_data=f"oldposts_add|{lnk[0]}|{channel_id}")])
        kb.append([InlineKeyboardButton("⬅️ Retour", callback_data="main_menu")])
        await query.edit_message_text("Sélectionne un lien pour ajouter ses anciens posts :", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("oldposts_add|"):
        link_id, channel_id = map(int, data.split("|")[1:])
        link = get_link(link_id)
        posts = []
        if link[1] == "Twitter":
            posts = await scrape_twitter(link[2], max_posts=POSTS_LIMIT)
        elif link[1] == "Instagram":
            posts = await scrape_instagram(link[2], max_posts=POSTS_LIMIT)
        add_old_posts(link_id, posts)
        await query.edit_message_text("Anciens posts ajoutés.")
    elif data == "main_menu":
        await show_main_menu(query)
    elif data == "test_posting":
        await test_posting(query, context)
    elif data == "menu_ads":
        await menu_ads(query)
    elif data.startswith("add_ad"):
        context.user_data["add_ad"] = True
        await query.edit_message_text("Envoie le texte de la pub suivi éventuellement de l'intervalle en heures (ex: 24)")
    elif data.startswith("delete_ad|"):
        ad_id = int(data.split("|")[1])
        delete_ad(ad_id)
        await menu_ads(query)
    elif data == "menu_admins":
        await menu_admins(query)
    elif data.startswith("add_admin"):
        context.user_data["add_admin"] = True
        await query.edit_message_text("Envoie l'identifiant Telegram à ajouter (numéro) ou transfère le contact à ajouter.")
    elif data.startswith("delete_admin|"):
        admin_id = int(data.split("|")[1])
        delete_admin(admin_id)
        await menu_admins(query)
    elif data == "assign_menu":
        await assign_menu(query)
    elif data.startswith("assign_chan|"):
        channel_id = int(data.split("|")[1])
        await assign_admins_to_channel_menu(query, channel_id)
    elif data.startswith("toggle_assign|"):
        channel_id, admin_id = map(int, data.split("|")[1:])
        current_admins = get_channel_admins(channel_id)
        if admin_id in current_admins:
            remove_admin_from_channel(admin_id, channel_id)
        else:
            assign_admin_to_channel(admin_id, channel_id)
        await assign_admins_to_channel_menu(query, channel_id)
    elif data == "noop":
        await query.answer()


async def menu_channels(query, context, user_id):
    channels = get_channels(admin_id=user_id if user_id != SUPERADMIN_ID else None)
    kb = []
    for chan in channels:
        try:
            ok, can_post, _ = await check_channel_access(context, chan[2])
            icon = "✅" if ok and can_post else "❌"
        except Exception:
            icon = "❓"
        kb.append([
            InlineKeyboardButton(f"{icon} {chan[1]} ({chan[2]})", callback_data="noop"),
            InlineKeyboardButton("🗑️", callback_data=f"delete_channel|{chan[0]}")
        ])
    kb.append([InlineKeyboardButton("➕ Ajouter un canal", callback_data="add_channel")])
    kb.append([InlineKeyboardButton("⬅️ Retour", callback_data="main_menu")])
    await query.edit_message_text("📋 Gestion des canaux :", reply_markup=InlineKeyboardMarkup(kb))


async def menu_links_select_channel(query, for_oldposts=False, user_id=None):
    channels = get_channels(admin_id=user_id if user_id and user_id != SUPERADMIN_ID else None)
    if not channels:
        await query.edit_message_text("Aucun canal enregistré.")
        return
    kb = []
    for chan in channels:
        if not for_oldposts:
            kb.append([InlineKeyboardButton(chan[1], callback_data=f"links_channel|{chan[0]}")])
        else:
            kb.append([InlineKeyboardButton(chan[1], callback_data=f"oldposts_links_channel|{chan[0]}")])
    kb.append([InlineKeyboardButton("⬅️ Retour", callback_data="main_menu")])
    await query.edit_message_text("Sélectionne un canal :", reply_markup=InlineKeyboardMarkup(kb))


async def menu_links(query, channel_id, user_id):
    links = get_links(channel_id)
    kb = [
        [InlineKeyboardButton(f"{lnk[1]} - {lnk[2]}", callback_data="noop"),
         InlineKeyboardButton("✏️", callback_data=f"edit_link|{lnk[0]}|{channel_id}"),
         InlineKeyboardButton("🗑️", callback_data=f"delete_link|{lnk[0]}|{channel_id}")]
        for lnk in links
    ]
    kb.append([InlineKeyboardButton("➕ Ajouter un lien", callback_data=f"add_link|{channel_id}")])
    kb.append([InlineKeyboardButton("⬅️ Retour", callback_data="menu_links")])
    await query.edit_message_text("🔗 Liens du canal :", reply_markup=InlineKeyboardMarkup(kb))


async def menu_admins(query):
    admins = get_admins()
    kb = []
    for aid in admins:
        kb.append([
            InlineKeyboardButton(f"👤 {aid}", callback_data="noop"),
            InlineKeyboardButton("🗑️", callback_data=f"delete_admin|{aid}") if aid != SUPERADMIN_ID else InlineKeyboardButton("🔒", callback_data="noop")
        ])
    kb.append([InlineKeyboardButton("➕ Ajouter un admin", callback_data="add_admin")])
    kb.append([InlineKeyboardButton("⬅️ Retour", callback_data="main_menu")])
    await query.edit_message_text("🛡 Gestion des admins :", reply_markup=InlineKeyboardMarkup(kb))


async def menu_ads(query):
    ads = get_ads()
    kb = []
    for ad in ads:
        display = ad[1][:20] + ("…" if len(ad[1]) > 20 else "")
        kb.append([
            InlineKeyboardButton(display, callback_data="noop"),
            InlineKeyboardButton("🗑️", callback_data=f"delete_ad|{ad[0]}")
        ])
    kb.append([InlineKeyboardButton("➕ Ajouter pub", callback_data="add_ad")])
    kb.append([InlineKeyboardButton("⬅️ Retour", callback_data="main_menu")])
    await query.edit_message_text("📢 Gestion des pubs :", reply_markup=InlineKeyboardMarkup(kb))


async def assign_menu(query):
    channels = get_channels()
    kb = []
    for chan in channels:
        kb.append([
            InlineKeyboardButton(f"{chan[1]}", callback_data=f"assign_chan|{chan[0]}")
        ])
    kb.append([InlineKeyboardButton("⬅️ Retour", callback_data="main_menu")])
    await query.edit_message_text("🔗 Choisis un canal à assigner :", reply_markup=InlineKeyboardMarkup(kb))


async def assign_admins_to_channel_menu(query, channel_id):
    all_admins = get_admins()
    current_admins = get_channel_admins(channel_id)
    kb = []
    for aid in all_admins:
        state = "✅" if aid in current_admins else "❌"
        kb.append([
            InlineKeyboardButton(f"{state} {aid}", callback_data=f"toggle_assign|{channel_id}|{aid}")
        ])
    kb.append([InlineKeyboardButton("⬅️ Retour", callback_data="assign_menu")])
    await query.edit_message_text(f"Admins assignés au canal {channel_id} :", reply_markup=InlineKeyboardMarkup(kb))


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        return
    text = update.message.text.strip()
    if context.user_data.get("add_admin"):
        try:
            admin_id = int(text)
            add_admin(admin_id, None)
            await update.message.reply_text(f"✅ Admin ajouté : {admin_id}")
        except Exception as e:
            await update.message.reply_text(f"❌ Erreur ajout admin: {e}")
        context.user_data.pop("add_admin", None)
        await show_main_menu(update)
        return
    if context.user_data.get("add_channel"):
        channel_id = None
        try:
            parts = text.split(maxsplit=1)
            if len(parts) == 2:
                name, chat_id = parts
            else:
                chat_id = parts[0]
                if chat_id.startswith("@"):  # username
                    name = chat_id[1:]
                elif "t.me/" in chat_id:
                    name = chat_id.split("/")[-1].split("?")[0]
                else:
                    name = chat_id
            ok, can_post, detail = await check_channel_access(context, chat_id)
            if not ok:
                logger.error("Access check failed for %s: %s", chat_id, detail)
                await update.message.reply_text(f"❌ Impossible de se connecter au canal/groupe : {detail}")
                return
            if not can_post:
                await update.message.reply_text("❌ Le bot n'est PAS admin dans ce canal/groupe !\nAjoute-le en admin puis réessaie.")
                return
            channel_id = add_channel(name, chat_id)
            assign_admin_to_channel(user_id, channel_id)
            await update.message.reply_text(f"✅ Canal '{name}' ajouté et connexion vérifiée (bot admin).")
            context.user_data.pop("add_channel")
            await show_main_menu(update)
            all_channels = get_channels(admin_id=user_id)
            for c in all_channels:
                if c[2] == chat_id:
                    await post_all_now(c[0], context, POSTS_LIMIT)
        except Exception as e:
            logger.exception("Erreur ajout canal")
            if channel_id:
                delete_channel(channel_id)
            await update.message.reply_text(
                f"❌ Erreur ajout canal: {e}\nFormat accepté :\n- MonCanal @moncanal\n- MonCanal -100xxxxxxxx\n- https://t.me/moncanal\nOu juste le lien ou l’identifiant seul."
            )
        return
    if context.user_data.get("add_link"):
        channel_id = context.user_data.pop("add_link")
        try:
            plat, url = text.split(maxsplit=1)
            plat = plat.capitalize()
            if plat not in ["Twitter", "Instagram"]:
                raise ValueError
            add_link(channel_id, plat, url)
            await update.message.reply_text(f"✅ Lien {plat} ajouté.")
            await show_main_menu(update)
            await post_all_now(channel_id, context, POSTS_LIMIT)
        except Exception:
            await update.message.reply_text("❌ Format: Twitter https://x.com/xxx")
        return
    if context.user_data.get("edit_link"):
        link_id, channel_id = context.user_data.pop("edit_link")
        update_link(link_id, text)
        await update.message.reply_text("✅ Lien modifié.")
        await show_main_menu(update)
        await post_all_now(channel_id, context, POSTS_LIMIT)
        return
    if context.user_data.get("add_ad"):
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            interval = int(parts[0])
            ad_text = parts[1]
        else:
            interval = 24
            ad_text = text
        add_ad(ad_text, interval)
        await update.message.reply_text("✅ Pub enregistrée.")
        await show_main_menu(update)
        return
    learn_ans = get_learn_answer(text)
    if learn_ans:
        await update.message.reply_text(learn_ans)
    elif AI_MODEL:
        reply = await asyncio.to_thread(ai_reply, user_id, text)
        await update.message.reply_text(reply)
    await show_main_menu(update)


### ========== POSTAGE ET NOTIF ==========

async def post_all_now(channel_id, context, limit=POSTS_LIMIT):
    links = get_links(channel_id)
    channel = get_channel(channel_id)
    if not channel:
        return
    chat_id = channel[2]
    ok, can_post, _ = await check_channel_access(context, chat_id)
    if not (ok and can_post):
        try:
            await context.bot.send_message(chat_id=SUPERADMIN_ID, text=f"❌ Le bot n'est pas admin sur le canal {chat_id}")
        except Exception:
            pass
        return
    for lnk in links:
        link_id, plat, url = lnk
        if plat == "Twitter":
            posts = await scrape_twitter(url, max_posts=limit)
        elif plat == "Instagram":
            posts = await scrape_instagram(url, max_posts=limit)
        else:
            continue
        for post in posts:
            if not post_exists(link_id, post):
                post_url = get_post_url(plat, url, post)
                await send_post_details(context, chat_id, plat, post_url)
                mark_posted(link_id, post)


async def scan_and_post(context: ContextTypes.DEFAULT_TYPE):
    links = get_links()
    for lnk in links:
        link_id, plat, url, channel_id = lnk
        channel = get_channel(channel_id)
        if not channel:
            continue
        chat_id = channel[2]
        ok, can_post, _ = await check_channel_access(context, chat_id)
        if not (ok and can_post):
            continue
        try:
            if plat == "Twitter":
                posts = await scrape_twitter(url, max_posts=POSTS_LIMIT)
            elif plat == "Instagram":
                posts = await scrape_instagram(url, max_posts=POSTS_LIMIT)
            else:
                continue
            has_new = False
            for post in posts:
                if not post_exists(link_id, post):
                    post_url = get_post_url(plat, url, post)
                    await send_post_details(context, chat_id, plat, post_url)
                    mark_posted(link_id, post)
                    has_new = True
            if not has_new:
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute('SELECT id, post_id FROM posts WHERE link_id = ? AND deja_poste = 0', (link_id,))
                old = c.fetchone()
                conn.close()
                if old:
                    post_db_id, post_id = old
                    post_url = get_post_url(plat, url, post_id)
                    await send_post_details(context, chat_id, plat, post_url)
                    set_posted(post_db_id)
        except Exception as e:
            logger.error(f"Erreur sur {plat} {url}: {e}")


async def scan_members_and_notify(context):
    channels = get_channels()
    for chan in channels:
        chat_id = chan[2]
        admins = get_channel_admins(chan[0])
        try:
            chat = await context.bot.get_chat(chat_id)
            count = chat.get_members_count() if hasattr(chat, "get_members_count") else None
            if count is None:
                count = await context.bot.get_chat_members_count(chat_id)
            last = get_last_stats(chan[0])
            if last is not None and count != last:
                diff = count - last
                txt = f"Le nombre de membres sur {chan[1]} ({chat_id}) a changé : {last} → {count} ({'+' if diff>0 else ''}{diff})"
                for admin_id in admins:
                    await context.bot.send_message(chat_id=admin_id, text=txt)
                if diff > 0:
                    welcome = get_welcome_text(chan[0])
                    if welcome:
                        await context.bot.send_message(chat_id=chat_id, text=welcome)
            set_last_stats(chan[0], count)
        except Exception as e:
            logger.error(f"Erreur stat membres {chat_id} : {e}")


async def send_due_ads(context):
    ads = due_ads()
    if not ads:
        return
    channels = get_channels()
    for ad_id, text, interval_hours, _ in ads:
        for chan in channels:
            chat_id = chan[2]
            ok, can_post, _ = await check_channel_access(context, chat_id)
            if ok and can_post:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=text)
                except Exception as e:
                    logger.error(f"send_due_ads failed on {chat_id}: {e}")
        mark_ad_sent(ad_id)


async def new_member_security(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type not in ("group", "supergroup"):
        return
    channel_id = update.effective_chat.id
    if not get_auto_ban(channel_id):
        return
    for member in update.message.new_chat_members:
        uname = member.username or ""
        if re.search(r"sex|porn|t\.me", uname, re.I):
            try:
                await update.effective_chat.ban_member(member.id)
                ban_user_record(channel_id, member.id, uname, "suspicious username")
            except Exception as e:
                logger.error("Auto-ban failed for %s on %s: %s", member.id, channel_id, e)


async def security_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type not in ("group", "supergroup"):
        return
    channel_id = update.effective_chat.id
    if not get_auto_ban(channel_id):
        return
    text = update.message.text or ""
    if re.search(r"http(s)?://t\.me/", text, re.I):
        try:
            await update.message.delete()
            user = update.effective_user
            await update.effective_chat.ban_member(user.id)
            ban_user_record(channel_id, user.id, user.username or "", "spam message")
        except Exception as e:
            logger.error("Auto-ban message failed for %s on %s: %s", user.id, channel_id, e)


def seconds_until_next_run():
    now = datetime.now()
    hours = [8, 12, 16, 20]
    times_today = [now.replace(hour=h, minute=0, second=0, microsecond=0) for h in hours if now.hour < h]
    if times_today:
        next_time = times_today[0]
    else:
        next_time = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
    while next_time.hour < 8:
        next_time += timedelta(hours=1)
    wait_seconds = (next_time - now).total_seconds()
    return max(10, int(wait_seconds))


async def scheduler_loop(app):
    while True:
        wait = seconds_until_next_run()
        logger.info(f"Prochain scan dans {wait//60} min.")
        await asyncio.sleep(wait)
        try:
            await scan_and_post(app)
            await scan_members_and_notify(app)
            await send_due_ads(app)
        except Exception as e:
            logger.error(f"Erreur sur scan_and_post ou scan_members_and_notify: {e}")


async def test_posting(query, context):
    channels = get_channels()
    results = []
    for chan in channels:
        try:
            ok, can_post, detail = await check_channel_access(context, chan[2])
            if not ok:
                results.append(f"{chan[1]} : ❌ Connexion impossible ({detail})")
            elif not can_post:
                results.append(f"{chan[1]} : ❌ Bot PAS admin")
            else:
                await context.bot.send_message(chat_id=chan[2], text="✅ Test d’envoi réussi !")
                results.append(f"{chan[1]} : ✅ OK")
        except Exception as e:
            results.append(f"{chan[1]} : ❌ {e}")
    msg = "\n".join(results) or "Aucun canal configuré."
    await query.edit_message_text("Résultat test d’envoi :\n" + msg)


### ========== MAIN ==========

async def main():
    init_db()
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("members", cmd_members))
    app.add_handler(CommandHandler("post", cmd_post))
    app.add_handler(CommandHandler("setwelcome", cmd_setwelcome))
    app.add_handler(CommandHandler("learn", cmd_learn))
    app.add_handler(CommandHandler("setsecurity", cmd_setsecurity))
    app.add_handler(CommandHandler("banned", cmd_banned))
    app.add_handler(CallbackQueryHandler(button))
    # Security filter should run before the general text handler
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, new_member_security))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, security_filter))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.ALL, always_main_menu))
    asyncio.create_task(scheduler_loop(app))
    await app.run_polling()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(main())
