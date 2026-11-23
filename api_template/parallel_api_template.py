# 并行 API 调用模板 - 使用多线程加速识别

import os
import requests
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

# API配置
API_URL = "https://llm.q2zy.com/v1/chat/completions"
API_KEY = os.environ.get("MUSO_API_KEY", "")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def recognize_single_image(image_path, max_retries=20, initial_delay=0.5, video_source="unknown"):
    """
    识别单张图片（带重试机制和指数退避）
    
    Args:
        image_path: 图片路径
        max_retries: 最大重试次数（增加到20次）
        initial_delay: 初始重试延迟（秒）
        video_source: 视频来源信息
    
    Returns:
        包含识别结果的字典，格式：
        {
            'answer': '字',
            'full_json': {...},  # 完整的JSON响应
            'success': True/False
        }
        失败返回None
    """
    image_base64 = image_to_base64(image_path)
    
    prompt = f"""任务：识别单帧书法图像中的汉字（该帧信息已足够唯一确定该字）。

上下文：
- 本图像是汉字书法教学视频的截图（source = {video_source}）。
- 当前帧包含足以判断目标汉字的可见笔画与结构。仅依据"已出现"的笔画与相对位置进行判断；不得臆测未出现的笔画。

输出规范（务必严格遵守）：
- 只输出一个 JSON 对象，且不包含任何多余文字、换行前缀或注释。
- 字段与取值：
  "answer": "<单个汉字>",                       // 最终判定（必须与概率列表首项一致）
  "probabilities": [                            // 按概率降序，长度 3–5
    {{"char":"<汉字1>","p":0.xx}},
    {{"char":"<汉字2>","p":0.xx}},
    {{"char":"<汉字3>","p":0.xx}}
  ],
  "rationale": "<推断依据>"

严格格式要求：
- 概率 p 为 0–1 之间的小数，保留两位；所有 p 之和必须等于 1.00（四舍五入后）。
- 若 "answer" ≠ probabilities[0].char，则令 "answer" = probabilities[0].char。
- 若可能的汉字数 < 3，用 null 代替汉字，概率则用 0.00 填充。
- 仅可输出中文汉字作为候选，不得输出词语、拼音或符号。

校准指引（内部执行，最终仅输出 JSON）：
- 先为 Top-K 候选分配相对打分 s_i ∈ {{1..9}}，再归一化为概率 p_i = s_i / Σ s_i。
- 若最高概率 < 0.50，仍给出最可能答案，并在 "rationale" 中简述歧义来源（例如"与'且'仅差左下短横"）。

现在输出符合上述规范的 JSON。"""
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "你是一个汉字识别助手"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "stream": False,
        "max_tokens": 300,  # 增加到300以适应JSON格式
        "temperature": 0.7
    }
    
    for retry in range(max_retries):
        try:
            # 增加超时时间到60秒
            response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # 清理可能的markdown代码块标记
                if content.startswith("```json"):
                    content = content[7:]  # 移除 ```json
                if content.startswith("```"):
                    content = content[3:]  # 移除 ```
                if content.endswith("```"):
                    content = content[:-3]  # 移除结尾的 ```
                content = content.strip()
                
                # 提取识别结果 - 解析 JSON 格式
                try:
                    # 尝试解析 JSON
                    recognition_data = json.loads(content)
                    
                    # 提取 answer 字段
                    if "answer" in recognition_data:
                        recognized_char = recognition_data["answer"]
                        # 返回完整信息
                        return {
                            'answer': recognized_char,
                            'full_json': recognition_data,
                            'success': True
                        }
                    else:
                        print(f"    警告: JSON 中缺少 'answer' 字段")
                        if retry < max_retries - 1:
                            delay = initial_delay * (2 ** retry)
                            time.sleep(min(delay, 10))
                            continue
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"    警告: JSON 解析失败 - {str(e)[:50]}")
                    print(f"    内容: {content[:100]}")
                    # 尝试重试而不是直接返回None
                    if retry < max_retries - 1:
                        delay = initial_delay * (2 ** retry)  # 指数退避
                        time.sleep(min(delay, 10))  # 最多等待10秒
                        continue
                    return None
            else:
                print(f"    重试 {retry + 1}/{max_retries}: 状态码 {response.status_code}")
                # 指数退避策略
                delay = initial_delay * (2 ** min(retry, 5))
                time.sleep(min(delay, 10))
                
        except requests.exceptions.Timeout:
            print(f"    重试 {retry + 1}/{max_retries}: 请求超时")
            delay = initial_delay * (2 ** min(retry, 5))
            time.sleep(min(delay, 10))
        except requests.exceptions.RequestException as e:
            print(f"    重试 {retry + 1}/{max_retries}: 网络错误 - {str(e)[:50]}")
            delay = initial_delay * (2 ** min(retry, 5))
            time.sleep(min(delay, 10))
        except Exception as e:
            print(f"    重试 {retry + 1}/{max_retries}: 未知错误 - {str(e)[:50]}")
            delay = initial_delay * (2 ** min(retry, 5))
            time.sleep(min(delay, 10))
    
    print(f"    ❌ 所有重试都失败了")
    return None


def recognize_image_parallel(image_path, num_attempts=10, max_workers=3, video_source="unknown"):
    """
    并行识别单张图片多次
    
    Args:
        image_path: 图片路径
        num_attempts: 识别次数
        max_workers: 并行线程数（降低到3以避免API限流）
        video_source: 视频来源信息
    
    Returns:
        识别统计结果字典，包含所有完整的JSON响应
    """
    print(f"\n正在并行识别: {image_path.name}")
    print(f"  并行线程数: {max_workers}, 总识别次数: {num_attempts}")
    
    recognition_counts = {}
    failed_count = 0
    all_json_responses = []  # 保存所有完整的JSON响应
    start_time = time.time()
    
    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，传递 video_source
        futures = [executor.submit(recognize_single_image, image_path, 20, 0.5, video_source) for _ in range(num_attempts)]
        
        # 收集结果
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result and result.get('success'):
                answer = result['answer']
                recognition_counts[answer] = recognition_counts.get(answer, 0) + 1
                all_json_responses.append({
                    'attempt': completed,
                    'answer': answer,
                    'full_response': result['full_json']
                })
                print(f"  进度: {completed}/{num_attempts} - ✅ 识别为: '{answer}'")
            else:
                failed_count += 1
                all_json_responses.append({
                    'attempt': completed,
                    'answer': None,
                    'full_response': None,
                    'error': 'Recognition failed'
                })
                print(f"  进度: {completed}/{num_attempts} - ❌ 识别失败")
    
    elapsed_time = time.time() - start_time
    success_rate = ((num_attempts - failed_count) / num_attempts) * 100
    
    # 统计结果
    print(f"\n  统计结果 (耗时: {elapsed_time:.2f}秒):")
    print(f"  成功率: {success_rate:.1f}% ({num_attempts - failed_count}/{num_attempts})")
    
    if recognition_counts:
        most_common_char = max(recognition_counts, key=recognition_counts.get)
        most_common_count = recognition_counts[most_common_char]
        
        for char, count in sorted(recognition_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / num_attempts) * 100
            print(f"    '{char}': {count}次 ({percentage:.1f}%)")
        
        return {
            'most_common_result': most_common_char,
            'most_common_count': most_common_count,
            'confidence': (most_common_count / num_attempts) * 100,
            'all_results': recognition_counts,
            'total_attempts': num_attempts,
            'successful_attempts': num_attempts - failed_count,
            'failed_attempts': failed_count,
            'success_rate': success_rate,
            'elapsed_time': elapsed_time,
            'all_json_responses': all_json_responses  # 添加所有完整的JSON响应
        }
    else:
        print(f"  ⚠️ 所有识别都失败了")
        return {
            'most_common_result': None,
            'total_attempts': num_attempts,
            'successful_attempts': 0,
            'failed_attempts': failed_count,
            'success_rate': 0,
            'elapsed_time': elapsed_time,
            'all_json_responses': all_json_responses  # 即使失败也保存响应
        }


def process_multiple_images_parallel(image_paths, num_attempts=10, max_workers=3, video_source="unknown"):
    """
    并行处理多张图片（每张图片内部也并行识别多次）
    
    Args:
        image_paths: 图片路径列表
        num_attempts: 每张图片的识别次数
        max_workers: 并行线程数（降低到3）
        video_source: 视频来源信息
    
    Returns:
        所有图片的识别结果列表
    """
    print(f"\n开始并行处理 {len(image_paths)} 张图片")
    print(f"每张图片识别 {num_attempts} 次，使用 {max_workers} 个并行线程\n")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    for image_path in image_paths:
        result = recognize_image_parallel(image_path, num_attempts, max_workers, video_source)
        # 总是保存结果，即使识别失败
        results.append({
            'image_path': str(image_path),
            'image_name': image_path.name,
            'recognition': result if result else {
                'most_common_result': None,
                'total_attempts': num_attempts,
                'successful_attempts': 0,
                'failed_attempts': num_attempts,
                'success_rate': 0,
                'all_json_responses': []
            }
        })
        print("-" * 60)
    
    total_time = time.time() - start_time
    print(f"\n全部完成！")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每张图片: {total_time/len(image_paths):.2f}秒")
    
    return results


# ============= 使用示例 =============

if __name__ == "__main__":
    # 示例1: 并行识别单张图片多次
    print("=" * 60)
    print("示例1: 单张图片并行识别")
    print("=" * 60)
    
    image_path = Path("../frames/包/frame_001.jpg")
    if image_path.exists():
        result = recognize_image_parallel(
            image_path, 
            num_attempts=10,    # 识别10次
            max_workers=3,      # 使用3个线程并行（降低以避免限流）
            video_source="包.mp4"  # 指定视频来源
        )
        
        if result:
            print(f"\n✅ 最终结果: '{result['most_common_result']}'")
            print(f"   置信度: {result['confidence']:.1f}%")
            print(f"   成功率: {result['success_rate']:.1f}%")
            print(f"   耗时: {result['elapsed_time']:.2f}秒")
    else:
        print(f"❌ 错误: 图片不存在 - {image_path}")
    
    print("\n" + "=" * 60)
    print("示例2: 多张图片批量并行处理")
    print("=" * 60)
    
    # 示例2: 处理一个文件夹中的多张图片
    frames_folder = Path("../frames/包")
    if frames_folder.exists():
        # 获取前3张图片进行测试
        image_files = sorted(list(frames_folder.glob("*.jpg")))[:3]
        
        if image_files:
            results = process_multiple_images_parallel(
                image_files,
                num_attempts=10,    # 每张图片识别10次
                max_workers=3,      # 使用3个线程并行（降低以避免限流）
                video_source="包.mp4"  # 指定视频来源
            )
            
            # 保存结果
            output_file = "parallel_test_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 结果已保存到: {output_file}")
        else:
            print(f"❌ 错误: 没有找到图片文件")
    else:
        print(f"❌ 错误: 文件夹不存在 - {frames_folder}")
