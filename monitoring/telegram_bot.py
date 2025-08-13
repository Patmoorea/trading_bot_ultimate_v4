def send_sentiment_alert(chat_id, text):
    from analysis.sentiment import analyze_with_finbert
    result = analyze_with_finbert(text)
    message = f"📊 Sentiment Analysis:\n{text}\n"
    message += f"Positive: {result['finbert'][0]:.2%}\n"
    message += f"Neutral: {result['finbert'][1]:.2%}\n"
    message += f"Negative: {result['finbert'][2]:.2%}"
    bot.send_message(chat_id, message)
