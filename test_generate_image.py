# test.py 경로로 들어가서 python test.py 실행
# FastAPI 서버 실행 후, http://localhost 8000 으로 접속하여 API 테스트 가능
# try it out 버튼을 누르면, description에 인풋 수정하고, 입력한 내용을 바탕으로 JD 생성 결과 확인 가능, 500 에러 안나는지 확인필요
# 200 status code 확인하고 생성된 내용 리뷰

# test result
'''
{
  "splitted": [
    "Generated Story: \n\n1. 'Developing effective lesson plans' - Every morning, Jenny, a passionate and talented chef, starts her day by carefully crafting lesson plans for her culinary class, ensuring they learn not only cutting techniques but also the art of food presentation.\n\n2. 'Evaluating students' progress' - By the afternoon, she is focusing on checking each student's progress, relishing the subtle improvement in their knife skills, and the newfound confidence they radiate in the kitchen.\n\n3. 'Creating engaging classroom environments' - To keep her students motivated, Jenny believes in setting an engaging classroom environment; hence she ends her day by decorating her kitchen classroom with seasonal produce and the aroma of freshly baked bread.\n\n4. 'Implementing innovative teaching methods' - She ends her week with a Friday \"Innovative Hour,\" where she introduces students to unconventional teaching methods, such as live cooking competitions and virtual reality-based culinary tours to keep their passion burning and always learning."
  ]
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
#--------------------------DALL E 이미지 생성 -----------------------------

# 요청 데이터 모델
class ImageRequest(BaseModel):
    prompt: str
    n: int = Field(default=1, ge=1, le=10)  # 1에서 10 사이의 값만 허용
    size: str = Field(default="1024x1024", pattern="^(1024x1024|512x512|256x256)$")
    quality: str = Field(default="standard", pattern="^(standard|hd)$")

# 응답 데이터 모델
class ImageResponse(BaseModel):
    image_urls: Optional[List[str]] = None
    error: Optional[str] = None

def generate_image(
    prompt: str,
    n: int = 1,
    size: str = "1024x1024",
    quality: str = "standard"
) -> Optional[List[str]]:
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            n=n,
            size=size,
            quality=quality,
            response_format="url"
        )
        image_urls = [data["url"] for data in response["data"]]
        return image_urls

    except openai.error.OpenAIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API 오류: {str(e)}"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"요청 오류: {str(e)}"
        )

@app.post("/generate-image", response_model=ImageResponse)
async def create_image(request: ImageRequest):
    try:
        image_urls = generate_image(
            prompt=request.prompt,
            n=request.n,
            size=request.size,
            quality=request.quality
        )
        if image_urls is None:
            raise HTTPException(
                status_code=500,
                detail="이미지 생성 실패"
            )
        return ImageResponse(image_urls=image_urls)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"예상치 못한 오류: {str(e)}"
        )

# 생성된 이미지 URL 캐시
image_cache = {}

# 캐시된 이미지 URL 조회
@app.get("/cached-image/{image_id}", response_model=ImageResponse)
async def get_cached_image(image_id: str):
    if image_id in image_cache:
        return ImageResponse(image_urls=image_cache[image_id])
    raise HTTPException(
        status_code=404,
        detail="이미지를 찾을 수 없습니다"
    )
    
import uvicorn
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
