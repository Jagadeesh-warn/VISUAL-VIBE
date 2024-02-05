import requests
import webbrowser
import time
from urllib import parse


# this class is designed to carried out the OAuth2 authorization flow
# and provide high level data requests for other scripts
class Oauth2:
    def __init__(self, consumer_key, consumer_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.token_endpoint = '/v2/oauth2/token'
        self.api_url = 'https://api.tumblr.com'

        # code request
        auth_url = 'https://www.tumblr.com/oauth2/authorize'
        auth_response = requests.get(auth_url, params={'client_id': self.consumer_key, 'response_type': 'code',
                                                       'scope': 'basic write offline_access', 'state': 100})
        # account access authorization
        webbrowser.open(auth_response.url)
        # paste here the redirected url
        url_with_code = input("Paste the previously opened web page URL: ").strip()
        # get the code parameter value
        code = parse.parse_qs(parse.urlparse(url_with_code).query)['code'][0]
        # response with the requested tokens
        code_grant_response = requests.post(self.api_url + self.token_endpoint,
                                            data={'grant_type': 'authorization_code', 'code': code,
                                                  'client_id': self.consumer_key,
                                                  'client_secret': self.consumer_secret})
        # raise an exception if something went wrong
        code_grant_response.raise_for_status()
        # store the current timestamp to check later if tokens are still valid
        access_token_timestamp = int(round(time.time()))
        # store the retrieved information in a dictionary called token
        # header will be passed to the headers param to make request
        cgr = code_grant_response.json()
        self.token = {'access_token': cgr['access_token'],
                      'expires_in': cgr['expires_in'],
                      'refresh_token': cgr['refresh_token'],
                      'time': access_token_timestamp,
                      'header': {
                          'Accept': 'application/json',
                          'Authorization': cgr['token_type'].capitalize() + ' ' + cgr['access_token']
                      }
                      }

    # Method used to refresh (if invalid) the access token and its extra data
    def refresh_tokens(self):
        curr_timestamp = int(round(time.time()))
        # check access token validity
        if curr_timestamp >= self.token['time'] + self.token['expires_in']:
            # invalid access token, request new one
            refresh_response = requests.post(self.api_url + self.token_endpoint,
                                             data={'grant_type': 'refresh_token',
                                                   'refresh_token': self.token['refresh_token'],
                                                   'client_id': self.consumer_key,
                                                   'client_secret': self.consumer_secret})
            refresh_response.raise_for_status()
            # get the response
            resp_json = refresh_response.json()
            # update access token data
            self.token['time'] = int(round(time.time()))
            self.token['access_token'] = resp_json['access_token']
            self.token['expires_in'] = resp_json['expires_in']
            self.token['refresh_token'] = resp_json['refresh_token']
            self.token['header']['Authorization'] = resp_json['token_type'].capitalize() + ' ' + resp_json[
                'access_token']

        return self.token

    # Method that can be used to get the results of a request to a specific end_point
    def query(self, end_point):
        self.refresh_tokens()
        response = requests.get(self.api_url + end_point, headers=self.token['header'])
        response.raise_for_status()
        return response.json()
