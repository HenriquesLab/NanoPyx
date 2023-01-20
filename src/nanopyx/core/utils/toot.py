def toot(msg):
    """Post a toot to Mastodon
    :param msg: The message to post
    """
    try:
        from mastodon import Mastodon
    except ImportError:
        print("Mastodon is not installed. Skipping toot.")
        return

    import requests

    r = requests.get('https://ipinfo.io')
    data = r.json()
    msg = f"[{data['city']}:{data['country']}]: {msg}"

    #   Set up Mastodon
    mastodon = Mastodon(
        access_token = 'qdFtF6ODx-z4w9O-GIu5iYUyddieCZeAUWXgyQ2scpE',
        api_base_url = 'https://botsin.space/'
    )

    mastodon.status_post(msg)
