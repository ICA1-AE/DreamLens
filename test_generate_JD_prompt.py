# test.py 경로로 들어가서 python test.py 실행
# FastAPI 서버 실행 후, http://localhost 8000 으로 접속하여 API 테스트 가능
# try it out 버튼을 누르면, description에 인풋 수정하고, 입력한 내용을 바탕으로 JD 생성 결과 확인 가능, 500 에러 안나는지 확인필요
# 200 status code 확인하고 생성된 내용 리뷰

#test result
'''
1. 'Developing effective lesson plans'\n2. 'Evaluating students' progress'\n3. 'Creating engaging classroom environments'\n4. 'Implementing innovative teaching methods
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
#--------------------------------- JD 생성 (test succeeded) ----------------------------------
# 요청 데이터 모델
class DescriptionRequest(BaseModel):
    job_title: str

# 응답 데이터 모델
class PromptResponse(BaseModel):
    prompt: Optional[str] = None
    error: Optional[str] = None

def generate_JD_prompt(job_title: str) -> Optional[str]:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""Create 4 Gerund phrases based on {job_title}'s 4 main job tasks specifically. 
                 Do not answer for the keyword not related to the job itself. \n # Output Format \n 1. '[mastering culinary arts] 2. ..."""},
                {"role": "user", "content": f"{job_title}"}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"ERROR!!: {e}")
        return None

@app.post("/generate-JD-prompt", response_model=PromptResponse)
async def create_prompt(request: DescriptionRequest):
    prompt = generate_JD_prompt(request.job_title)
    
    if prompt is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate prompt"
        )
    
    return PromptResponse(prompt=prompt)

import uvicorn
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
