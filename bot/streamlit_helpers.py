import streamlit as st
import asyncio
import nest_asyncio
import os
import logging

logger = logging.getLogger(__name__)

from datetime import datetime, timezone

from bot.core import TradingBotM4
from src.bot.utils import StreamlitSessionManager
from src.notifications.telegram_bot import TelegramBot


session_manager = StreamlitSessionManager()

# Ajoute les autres imports nécessaires (logger, session_manager, etc.)


def get_bot():
    if "bot_instance" in st.session_state and st.session_state.bot_instance is not None:
        return st.session_state.bot_instance

    try:
        session_manager.protect_session()
        logger.info("Creating new bot instance...")
        bot = TradingBotM4()

        async def initialize_bot():
            try:
                await bot.async_init()
                # Vérifie que .start() existe AVANT d’appeler
                if getattr(bot, "ws_manager", None) is not None:
                    if not await bot.start():
                        raise Exception("Bot initialization failed")
                else:
                    logger.info(
                        "WebSocket manager non initialisé (testnet ou mode restreint), skip .start()"
                    )
                bot._initialized = True
                logger.info("Bot initialized successfully STREAM")
                return bot
            except Exception as init_error:
                logger.error(f"Bot initialization error: {init_error}")
                raise

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        nest_asyncio.apply()

        try:
            if loop.is_running():
                raise RuntimeError(
                    "Impossible d'initialiser le bot dans le cache_resource de Streamlit en mode async pur. Solution : initialiser dans une fonction async native."
                )
            else:
                bot = loop.run_until_complete(initialize_bot())
        except RuntimeError as e:
            logger.error(f"RuntimeError during bot init: {e}")
            bot = loop.run_until_complete(initialize_bot())

        if not bot or not getattr(bot, "_initialized", False):
            raise Exception("Bot initialization incomplete")

        st.session_state.bot_instance = bot
        logger.info(
            f"Bot ready - Status: {bot.ws_connection.get('status', 'initializing')} - Mode: {getattr(bot, 'trading_mode', 'production')}"
        )

        return bot
    except Exception as e:
        logger.error(f"Bot creation failed: {e}")
        if "bot_instance" in st.session_state:
            del st.session_state.bot_instance
        if "loop" in st.session_state:
            del st.session_state.loop
        return None
