# test.py 경로로 들어가서 python test.py 실행
# FastAPI 서버 실행 후, http://localhost 8000 으로 접속하여 API 테스트 가능
# try it out 버튼을 누르면, description에 인풋 수정하고, 입력한 내용을 바탕으로 JD 생성 결과 확인 가능, 500 에러 안나는지 확인필요
# 200 status code 확인하고 생성된 내용 리뷰

# test result
'''
curl -X 'POST' \
  'http://localhost:8000/split-stories' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "stories": [
    "Generated Story: \n\n1. '\''Developing effective lesson plans'\'' - Every morning, Jenny, a passionate and talented chef, starts her day by carefully crafting lesson plans for her culinary class, ensuring they learn not only cutting techniques but also the art of food presentation.\n
    \n2. '\''Evaluating students'\'' progress'\'' - By the afternoon, she is focusing on checking each student'\''s progress, relishing the subtle improvement in their knife skills, and the newfound confidence they radiate in the kitchen.\n
    \n3. '\''Creating engaging classroom environments'\'' - To keep her students motivated, Jenny believes in setting an engaging classroom environment; hence she ends her day by decorating her kitchen classroom with seasonal produce and the aroma of freshly baked bread.\n
    \n4. '\''Implementing innovative teaching methods'\'' - She ends her week with a Friday \"Innovative Hour,\" where she introduces students to unconventional teaching methods, such as live cooking competitions and virtual reality-based culinary tours to keep their passion burning and always learning.""
  ]
}'

이게 나뉜건가? 
test 리스트 만들어보면 에러뜸..
'''

import openai
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    openai.api_key = api_key
else:
    print("API key not found. Please set the OPENAI_API_KEY environment variable.")

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import openai
from typing import Optional

import requests

# FastAPI 앱 인스턴스 생성
app = FastAPI()
#--------------------------생성된 스토리 split 처리  ----------------------------- 

# 요청 데이터 모델
class StoriesRequest(BaseModel):
    stories: List[str]

# 응답 데이터 모델
class StoriesResponse(BaseModel):
    splitted: List[str]

def split_stories(stories: List[str]) -> List[str]:
    try:
        return [story.split(':')[1].strip() for story in stories[0].split('\n\n')]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error extracting stories: {str(e)}"
        )

@app.post("/split-stories", response_model=StoriesResponse)
async def create_stories(request: StoriesRequest):
    try:
        result = split_stories(request.stories)
        return StoriesResponse(splitted=result)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
    
import uvicorn
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
