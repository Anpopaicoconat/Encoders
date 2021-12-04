from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

with open('token.txt', 'r') as token:
    TOKEN = token.read()
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
    
@dp.message_handler(content_types=['text'])
async def get_text_messages(msg: types.Message):
    inp = msg.text
    await msg.answer(inp)
      
if __name__ == '__main__':
   executor.start_polling(dp)
