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
    JD_prompts: Optional[str] = None
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
    JD_prompts = generate_JD_prompt(request.job_title)
    
    if JD_prompts is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate JD_prompts"
        )
    
    return PromptResponse(prompt=JD_prompts)


#--------------------------JD 기반 4개 연이어진 하루 일상 스토리 생성(test succeeded) ----------------------------- 

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

#--------------------------생성된 스토리 split 처리 (test completed) ----------------------------- 

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

#--------------------------DALL E 프롬프트 생성 (test completed) -----------------------------
import ast
from typing import Optional, List

app = FastAPI()

# 요청 데이터 모델
class PromptRequest(BaseModel):
    splitted: List[str] 

# 응답 데이터 모델
class PromptResponse(BaseModel):
    image_prompts: Optional[List[str]] = None
    error: Optional[str] = None

def convert_to_image_prompts(splitted: str) -> Optional[List[str]]:
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
                {"role": "user", "content": f"{splitted}"}
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

#--------------------------DALL E 이미지 생성 (test completed)-----------------------------

# 요청 데이터 모델
class ImageRequest(BaseModel):
    prompt: str # !!! 이전 아웃풋이 리스트 형태여서 각 인덱스별로 반복으로 받아야함
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

#--------------------------DALL E 이미지 출력-----------------------------


# 요청 데이터 모델
class ImagePromptRequest(BaseModel):
    img_prompts: List[str]

# 응답 데이터 모델
class ImageGenerationResult(BaseModel):
    prompt_number: int
    url: Optional[str] = None
    status: str

class ImageResponse(BaseModel):
    results: List[ImageGenerationResult]

def print_image(prompt: str, n: int = 1) -> Optional[List[str]]:
    try:
        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            n=n,
            size="1024x1024",
            quality="standard",
            response_format="url"
        )
        return [data["url"] for data in response["data"]]
    except Exception as e:
        print(f"이미지 생성 오류: {e}")
        return None

def print_images(img_prompts: List[str]) -> List[ImageGenerationResult]:
    results = []
    
    for i, prompt in enumerate(img_prompts, 1):
        image_urls = generate_image(prompt, n=1)
        if image_urls:
            results.append(
                ImageGenerationResult(
                    prompt_number=i,
                    url=image_urls[0],
                    status="success"
                )
            )
        else:
            results.append(
                ImageGenerationResult(
                    prompt_number=i,
                    status="failed"
                )
            )
    
    return results

@app.post("/generate-images", response_model=ImageResponse)
async def create_images(request: ImagePromptRequest):
    try:
        results = print_images(request.img_prompts)
        return ImageResponse(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"이미지 생성 중 오류 발생: {str(e)}"
        )

# 결과 캐시
results_cache = {}

# 캐시된 결과 조회
@app.get("/cached-results/{request_id}", response_model=ImageResponse)
async def get_cached_results(request_id: str):
    if request_id in results_cache:
        return ImageResponse(results=results_cache[request_id])
    raise HTTPException(
        status_code=404,
        detail="결과를 찾을 수 없습니다"
    )

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

import uvicorn
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)