import json
import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import openai
from typing import Optional

from starlette.middleware.cors import CORSMiddleware

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 환경 변수 설정
load_dotenv()

#--------------------------------- JD 생성 (test succeeded) ----------------------------------
# 요청 데이터 모델
class DescriptionRequest(BaseModel):
    job_title: str

# 응답 데이터 모델
class JobActions(BaseModel):
    actions: List[str]
    job_title: str

ACTION_PROMPT = """
입력받은 직업에서 일반적으로 수행하는 4가지 구체적인 행동을 JSON 형식으로 출력하세요. 
JSON의 키는 "job_title"과 "actions"로 구성되며, "actions"는 각 행동을 문자열로 포함하는 배열입니다.

예시:
입력된 직업명: "경찰관"
출력:
{
  "job_title": "경찰관",
  "actions": [
    "교통 체증을 완화하기 위해 교차로에서 차량을 통제합니다.",
    "긴급 상황에 대응하여 사건 현장으로 출동합니다.",
    "범죄 용의자를 추적하고 체포합니다.",
    "시민들의 안전을 위해 지역 순찰을 수행합니다."
  ]
}

입력된 직업명: "소방관"
출력:
{
  "job_title": "소방관",
  "actions": [
    "화재가 발생한 건물에서 사람들을 구조합니다.",
    "화재 진압을 위해 고압 호스를 사용합니다.",
    "긴급 구조 훈련을 통해 팀워크를 강화합니다.",
    "사고 현장에서 부상자를 응급처치합니다."
  ]
}
"""

def generate_job_actions(job_title: str) -> JobActions:
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ACTION_PROMPT},
                {"role": "user", "content": f"{job_title}"}
            ]
        )
        print(json.loads(response.choices[0].message.content))
        parsed_data = json.loads(response.choices[0].message.content)
        job_actions = JobActions(**parsed_data)
        return job_actions
    except Exception as e:
        print(f"ERROR!!: {e}")
        return None

@app.post("/generate-job-actions", response_model=JobActions)
async def create_prompt(request: DescriptionRequest) -> JobActions:
    job_actions = generate_job_actions(request.job_title)
    
    if job_actions is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate JD_prompts"
        )
    
    return job_actions


#--------------------------JD 기반 4개 연이어진 하루 일상 스토리 생성(test succeeded) ----------------------------- 

# 요청 데이터 모델
class StoryRequest(BaseModel):
    job_actions: List[str]
    user_name: str
    genre: str
    job_title: str

# 응답 데이터 모델
class StoryResponse(BaseModel):
    story: str

# 전역 stories 리스트
stories = []

def generate_story(request: StoryRequest) ->str:
    prompt = f"""
        당신은 창의적이고 이야기 제작에 능숙한 AI 작가입니다. 
        사용자가 선택한 직업의 행동을 활용해서 장르에 맞는 스토리를 만들어 주세요. 
        아래는 스토리를 작성하기 위한 조건입니다:

        입력 정보:
        name: {request.user_name}
        genre: {request.genre}
        job_title: {request.job_title}
        직업 행동:
        {request.job_actions}

        요구 사항:
        사용자의 이름과 선택한 장르를 반영하여 스토리를 만드세요.
        직업 행동은 스토리 속에서 자연스럽게 녹아들도록 하되, 각 캡션이 끝날 때 [image]를 추가하여 사진 위치를 표시하세요.
        스토리는 논리적이고 흥미롭게 진행되어야 합니다.

        입력 예시:
        name: 민준
        genre: 판타지
        job_title: 탐험가
        직업 행동: ["숲 속의 고대 유적", "마법의 빛을 발하는 수정 구슬", "전설의 용과의 조우"]
        출력 예시:
        민준은 어릴 적부터 고대 유적을 탐험하는 꿈을 꿨다. 어느 날, 그는 신비로운 숲으로 떠나기로 결심했다. 숲 깊은 곳에서 그는 오랜 시간 잊혀졌던 고대 유적을 발견했다. 유적은 초록 이끼로 뒤덮여 있었지만, 그곳에 서린 고대의 힘은 여전히 강렬했다. [image]
        그가 유적 안으로 들어서자, 빛나는 수정 구슬이 공중에 떠올라 그의 주위를 감쌌다. 구슬은 그의 손길에 반응하며 은은한 빛을 내뿜었다. 이는 분명히 고대 마법의 유물임에 틀림없었다. [image]
        구슬의 빛을 따라가던 민준은 전설로만 전해지던 용과 마주쳤다. 용은 그의 방문을 기다렸다는 듯이 웅장한 목소리로 말했다. "드디어, 내가 선택한 인간이 왔구나." [image]
        이제 민준의 모험은 막 시작되었다.
        """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 뛰어난 소설가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating story: {str(e)}"
        )

@app.post("/generate-story", response_model=StoryResponse)
async def create_story(request: StoryRequest)->StoryResponse:
    try:
        generated_story = generate_story(request)
        return StoryResponse(story=generated_story)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# # 모든 스토리 조회
# @app.get("/stories", response_model=List[str])
# async def get_stories():
#     return stories
#
# # 스토리 초기화
# @app.post("/clear-stories")
# async def clear_stories():
#     stories.clear()
#     return {"message": "Stories cleared successfully"}

#--------------------------생성된 스토리 split 처리 (test completed) ----------------------------- 

# # 요청 데이터 모델
# class StoriesRequest(BaseModel):
#     stories: List[str]
#
# # 응답 데이터 모델
# class StoriesResponse(BaseModel):
#     splitted: List[str]
#
# def split_stories(stories: List[str]) -> List[str]:
#     try:
#         return [story.split(':')[1].strip() for story in stories[0].split('\n\n')]
#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Error extracting stories: {str(e)}"
#         )
#
# @app.post("/split-stories", response_model=StoriesResponse)
# async def create_stories(request: StoriesRequest):
#     try:
#         result = split_stories(request.stories)
#         return StoriesResponse(splitted=result)
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )

#--------------------------DALL E 프롬프트 생성 (test completed) -----------------------------
# import ast
# from typing import Optional, List
#
# app = FastAPI()
#
# # 요청 데이터 모델
# class PromptRequest(BaseModel):
#     splitted: List[str]
#
# # 응답 데이터 모델
# class PromptResponse(BaseModel):
#     image_prompts: Optional[List[str]] = None
#     error: Optional[str] = None
#
# def convert_to_image_prompts(splitted: str) -> Optional[List[str]]:
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": """Convert the following job descriptions into suitable DALL-E image prompts.
#                   #Steps
#                   ##1. Extract a few specific scenery and action details about the main character##.
#                   Example : 'A woman visiting an organic dairy farm and tasting different cheeses'
#                   ##2. Make the background and mood for the all results same as 'charming, cozy watercolor illustration on white background illustration on white background' ##
#                   ##3. Describe a detailed outfit and appearance of the main character that fits on the job. Make the outfit and appearance for the all results same.##.
#                   ##4. Add Seed number for the all results same.##.
#
#                  #Notes
#                  ##Output: Only print raw text prompt.##
#                   #Output Format Example
#                   '`A woman wearing a white apron and a chef's hat visiting an organic dairy farm and tasting different cheeses, charming, cozy watercolor illustration on white background illustration on white background. Seed number: 1234'
#                  `'
#                  #Output :
#                  """},
#                 {"role": "user", "content": f"{splitted}"}
#             ]
#         )
#         image_prompts = response.choices[0].message['content']
#
#         # ast.literal_eval을 사용하여 문자열을 리스트로 변환
#         try:
#             # 리스트 형태가 아닌 경우 대괄호로 감싸서 리스트로 만듦
#             if not (image_prompts.startswith('[') and image_prompts.endswith(']')):
#                 image_prompts = f"[{image_prompts}]"
#             return ast.literal_eval(image_prompts)
#         except (ValueError, SyntaxError) as e:
#             # 변환에 실패한 경우 단일 항목 리스트로 반환
#             return [image_prompts]
#
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error generating image prompt: {str(e)}"
#         )
#
# @app.post("/convert-prompt", response_model=PromptResponse)
# async def create_image_prompt(request: PromptRequest):
#     try:
#         converted_prompts = convert_to_image_prompts(request.prompts)
#         if converted_prompts is None:
#             raise HTTPException(
#                 status_code=500,
#                 detail="Failed to generate image prompts"
#             )
#         return PromptResponse(image_prompts=converted_prompts)
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )
#
# # 결과 캐시를 위한 딕셔너리
# prompt_cache = {}
#
# # 캐시된 결과 조회
# @app.get("/cached-prompt/{prompt_id}", response_model=PromptResponse)
# async def get_cached_prompt(prompt_id: str):
#     if prompt_id in prompt_cache:
#         return PromptResponse(image_prompts=prompt_cache[prompt_id])
#     raise HTTPException(
#         status_code=404,
#         detail="Prompt not found in cache"
#     )
#
# #--------------------------DALL E 이미지 생성 (test completed)-----------------------------
#
# # 요청 데이터 모델
# class ImageRequest(BaseModel):
#     prompt: str # !!! 이전 아웃풋이 리스트 형태여서 각 인덱스별로 반복으로 받아야함
#     n: int = Field(default=1, ge=1, le=10)  # 1에서 10 사이의 값만 허용
#     size: str = Field(default="1024x1024", pattern="^(1024x1024|512x512|256x256)$")
#     quality: str = Field(default="standard", pattern="^(standard|hd)$")
#
# # 응답 데이터 모델
# class ImageResponse(BaseModel):
#     image_urls: Optional[List[str]] = None
#     error: Optional[str] = None
#
# def generate_image(
#     prompt: str,
#     n: int = 1,
#     size: str = "1024x1024",
#     quality: str = "standard"
# ) -> Optional[List[str]]:
#     try:
#         response = openai.Image.create(
#             model="dall-e-3",
#             prompt=prompt,
#             n=n,
#             size=size,
#             quality=quality,
#             response_format="url"
#         )
#         image_urls = [data["url"] for data in response["data"]]
#         return image_urls
#
#     except openai.error.OpenAIError as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"OpenAI API 오류: {str(e)}"
#         )
#
# @app.post("/generate-image", response_model=ImageResponse)
# async def create_image(request: ImageRequest):
#     try:
#         image_urls = generate_image(
#             prompt=request.prompt,
#             n=request.n,
#             size=request.size,
#             quality=request.quality
#         )
#         if image_urls is None:
#             raise HTTPException(
#                 status_code=500,
#                 detail="이미지 생성 실패"
#             )
#         return ImageResponse(image_urls=image_urls)
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"예상치 못한 오류: {str(e)}"
#         )
#
# # 생성된 이미지 URL 캐시
# image_cache = {}
#
# # 캐시된 이미지 URL 조회
# @app.get("/cached-image/{image_id}", response_model=ImageResponse)
# async def get_cached_image(image_id: str):
#     if image_id in image_cache:
#         return ImageResponse(image_urls=image_cache[image_id])
#     raise HTTPException(
#         status_code=404,
#         detail="이미지를 찾을 수 없습니다"
#     )
#
# #--------------------------DALL E 이미지 출력-----------------------------
#
#
# # 요청 데이터 모델
# class ImagePromptRequest(BaseModel):
#     img_prompts: List[str]
#
# # 응답 데이터 모델
# class ImageGenerationResult(BaseModel):
#     prompt_number: int
#     url: Optional[str] = None
#     status: str
#
# class ImageResponse(BaseModel):
#     results: List[ImageGenerationResult]
#
# def print_image(prompt: str, n: int = 1) -> Optional[List[str]]:
#     try:
#         response = openai.Image.create(
#             model="dall-e-3",
#             prompt=prompt,
#             n=n,
#             size="1024x1024",
#             quality="standard",
#             response_format="url"
#         )
#         return [data["url"] for data in response["data"]]
#     except Exception as e:
#         print(f"이미지 생성 오류: {e}")
#         return None
#
# def print_images(img_prompts: List[str]) -> List[ImageGenerationResult]:
#     results = []
#
#     for i, prompt in enumerate(img_prompts, 1):
#         image_urls = generate_image(prompt, n=1)
#         if image_urls:
#             results.append(
#                 ImageGenerationResult(
#                     prompt_number=i,
#                     url=image_urls[0],
#                     status="success"
#                 )
#             )
#         else:
#             results.append(
#                 ImageGenerationResult(
#                     prompt_number=i,
#                     status="failed"
#                 )
#             )
#
#     return results
#
# @app.post("/generate-images", response_model=ImageResponse)
# async def create_images(request: ImagePromptRequest):
#     try:
#         results = print_images(request.img_prompts)
#         return ImageResponse(results=results)
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"이미지 생성 중 오류 발생: {str(e)}"
#         )
#
# # 결과 캐시
# results_cache = {}
#
# # 캐시된 결과 조회
# @app.get("/cached-results/{request_id}", response_model=ImageResponse)
# async def get_cached_results(request_id: str):
#     if request_id in results_cache:
#         return ImageResponse(results=results_cache[request_id])
#     raise HTTPException(
#         status_code=404,
#         detail="결과를 찾을 수 없습니다"
#     )

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

import uvicorn
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8600)