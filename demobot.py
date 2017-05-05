from slackbot.bot import Bot, respond_to
import logging
import re
import requests
import html
import slackbot_settings as settings

'''
TODO: Check for any Pre-train model for Keras
@respond_to('(.*)', re.IGNORECASE)
def hi(message, phrase):
    message.reply('I can understand hi or HI!')
    # react with thumb up emoji
    message.react('+1')
'''

@respond_to('I love plantain')
def love(message):
    message.reply('me too :heart:')


def main():
    logging.basicConfig(level=logging.DEBUG)
    bot = Bot()
    bot.run()


if __name__ == "__main__":
    main()
