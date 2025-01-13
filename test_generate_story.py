# test.py 경로로 들어가서 python test.py 실행
# FastAPI 서버 실행 후, http://localhost 8000 으로 접속하여 API 테스트 가능
# try it out 버튼을 누르면, description에 인풋 수정하고, 입력한 내용을 바탕으로 JD 생성 결과 확인 가능, 500 에러 안나는지 확인필요
# 200 status code 확인하고 생성된 내용 리뷰

# test result
'''
{
  "stories": [
    "Generated Story: \n\n1. 'Developing effective lesson plans' - Every morning, Jenny, a passionate and talented chef, starts her day by carefully crafting lesson plans for her culinary class, ensuring they learn not only cutting techniques but also the art of food presentation.\n
    \n2. 'Evaluating students' progress' - By the afternoon, she is focusing on checking each student's progress, relishing the subtle improvement in their knife skills, and the newfound confidence they radiate in the kitchen.\n
    \n3. 'Creating engaging classroom environments' - To keep her students motivated, Jenny believes in setting an engaging classroom environment; hence she ends her day by decorating her kitchen classroom with seasonal produce and the aroma of freshly baked bread.\n
    \n4. 'Implementing innovative teaching methods' - She ends her week with a Friday \"Innovative Hour,\" where she introduces students to unconventional teaching methods, such as live cooking competitions and virtual reality-based culinary tours to keep their passion burning and always learning."
  ],
  "error": null
}
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
#--------------------------JD 기반 4개 연이어진 하루 일상 스토리 생성 (test succeeded) ----------------------------- 
# 요청 데이터 모델
class StoryRequest(BaseModel):
    JD_prompts: str
    user_name: str
    job_title: str

# 응답 데이터 모델
class StoryResponse(BaseModel):
    stories: List[str]
    error: Optional[str] = None

# 전역 stories 리스트
stories = []

def generate_story(JD_prompts: str, user_name: str, job_title: str) -> List[str]:
    if not JD_prompts:
        raise ValueError("Prompt is not available")
        
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"""
                 Create 4 sequencial stories about {user_name}'s daily routine based on the job task in {JD_prompts}. 
                 Assume {user_name} fullfilled the dream to be {job_title}. 
                 \n #Steps
                 \n ##Generate Daily Routine##.: Create a fun story based on the job and its duties . 
                 Each story should include the character's activities in {JD_prompts} sequencially, and should not exceed 4 sentences. 
                 And  \n
                 \n #Output Format\nGenerated Story: `[Short story within 4 sentences]`
                 """},
                {"role": "user", "content": f"{JD_prompts}"}
            ]
        )
        story = response.choices[0].message['content']
        stories.append(story)
        return stories
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating story: {str(e)}"
        )

@app.post("/generate-story", response_model=StoryResponse)
async def create_story(request: StoryRequest):
    try:
        generated_stories = generate_story(
            prompts=request.prompts,
            user_name=request.user_name,
            job_title=request.job_title
        )
        return StoryResponse(stories=generated_stories)
    except ValueError as ve:
        return StoryResponse(stories=[], error=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# 모든 스토리 조회
@app.get("/stories", response_model=List[str])
async def get_stories():
    return stories

# 스토리 초기화
@app.post("/clear-stories")
async def clear_stories():
    stories.clear()
    return {"message": "Stories cleared successfully"}

import uvicorn
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
