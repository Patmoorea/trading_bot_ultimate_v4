def send_news_alert(news_item):
    """Envoie les alertes news avec score de sentiment"""
    msg = f"📰 {news_item['source']}\nScore: {news_item['score']:.2f}"
    bot.send_message(CHAT_ID, msg)
