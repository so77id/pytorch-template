import slack
import os

def send_slack(text):
    slack_token = "xoxp-179551653108-179513861619-644091832372-d6cc846c3b712ffc2aa6994ac913f0d2"
    slack_channel = "#mr-fer-experiments"

    client = slack.WebClient(token=slack_token)

    response = client.chat_postMessage(
        channel=slack_channel,
        text=text)
