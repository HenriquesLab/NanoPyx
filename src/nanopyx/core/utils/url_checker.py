import requests


def url_checker(url):
    """
    Check the reachability of a URL by sending an HTTP GET request.

    Args:
        url (str): The URL to be checked.

    Returns:
        str: A message indicating whether the URL is reachable or not.

    Raises:
        SystemExit: If an exception occurs during the HTTP request, the function exits with an error message.

    Example:
        result = url_checker("https://www.example.com")
        # Output: "https://www.example.com: is reachable"

    Note:
        This function sends an HTTP GET request to the specified URL and checks the HTTP status code.
        If the status code is 200, it indicates that the URL is reachable.
        If there are any exceptions during the request, it raises a SystemExit with an error message.
    """
    try:
        # Get Url
        get = requests.get(url)
        # if the request succeeds
        if get.status_code == 200:
            return f"{url}: is reachable"
        else:
            return f"{url}: is Not reachable, status_code: {get.status_code}"

    # Exception
    except requests.exceptions.RequestException as e:
        # print URL with Errs
        raise SystemExit(f"{url}: is Not reachable \nErr: {e}")
