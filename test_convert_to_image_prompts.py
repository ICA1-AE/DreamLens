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
#--------------------------DALL E 프롬프트 생성 -----------------------------
import ast
from typing import Optional, List

app = FastAPI()

# 요청 데이터 모델
class PromptRequest(BaseModel):
    prompts: str

# 응답 데이터 모델
class PromptResponse(BaseModel):
    image_prompts: Optional[List[str]] = None
    error: Optional[str] = None

def convert_to_image_prompts(prompts: str) -> Optional[List[str]]:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """Convert the following job descriptions into suitable DALL-E image prompts.
                  #Steps
                  ##1. Extract a few specific scenery and action details about the main character##. 
                  Example : 'A woman visiting an organic dairy farm and tasting different cheeses'
                  ##2. Make the background and mood for the all results same as 'charming, cozy watercolor illustration on white background illustration on white background' ##
                  ##3. Describe a detailed outfit and appearance of the main character that fits on the job. Make the outfit and appearance for the all results same.##. 
                  ##4. Add Seed number for the all results same.##.
                
                 #Notes 
                 ##Output: Only print raw text prompt.##
                  #Output Format Example
                  '`A woman wearing a white apron and a chef's hat visiting an organic dairy farm and tasting different cheeses, charming, cozy watercolor illustration on white background illustration on white background. Seed number: 1234'
                 `'
                 #Output :
                 """},
                {"role": "user", "content": f"{prompts}"}
            ]
        )
        image_prompts = response.choices[0].message['content']
        
        # ast.literal_eval을 사용하여 문자열을 리스트로 변환
        try:
            # 리스트 형태가 아닌 경우 대괄호로 감싸서 리스트로 만듦
            if not (image_prompts.startswith('[') and image_prompts.endswith(']')):
                image_prompts = f"[{image_prompts}]"
            return ast.literal_eval(image_prompts)
        except (ValueError, SyntaxError) as e:
            # 변환에 실패한 경우 단일 항목 리스트로 반환
            return [image_prompts]
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating image prompt: {str(e)}"
        )

@app.post("/convert-prompt", response_model=PromptResponse)
async def create_image_prompt(request: PromptRequest):
    try:
        converted_prompts = convert_to_image_prompts(request.prompts)
        if converted_prompts is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate image prompts"
            )
        return PromptResponse(image_prompts=converted_prompts)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# 결과 캐시를 위한 딕셔너리
prompt_cache = {}

# 캐시된 결과 조회
@app.get("/cached-prompt/{prompt_id}", response_model=PromptResponse)
async def get_cached_prompt(prompt_id: str):
    if prompt_id in prompt_cache:
        return PromptResponse(image_prompts=prompt_cache[prompt_id])
    raise HTTPException(
        status_code=404,
        detail="Prompt not found in cache"
    )
    
import uvicorn
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)