import asyncio
import random
from itertools import cycle

from openai import AsyncOpenAI


def _extract_usage(resp, is_gpt5: bool):
    """Normalize token-usage fields across Chat Completions and Responses APIs."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None

    if is_gpt5:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    else:
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


async def get_completion_async(
    client: AsyncOpenAI,
    model_name: str,
    prompt,
    return_reasoning: bool,
    return_usage: bool,
    max_retry: int,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    semaphore: asyncio.Semaphore,
):
    """Call one prompt with retry/backoff under a per-client concurrency limit."""
    async with semaphore:
        backoff = 1.5
        is_gpt5 = model_name.startswith("gpt-5")

        for attempt in range(1, max_retry + 1):
            try:
                if is_gpt5:
                    resp = await client.responses.create(
                        model=model_name,
                        input=prompt,
                        max_output_tokens=max_completion_tokens,
                    )

                    output_text = resp.output_text or ""
                    assert len(output_text) > 10
                    usage = _extract_usage(resp, is_gpt5=True)

                    if return_reasoning:
                        reasoning_text = ""
                        rc = getattr(resp, "reasoning_content", None)
                        if rc:
                            reasoning_text = "".join(
                                getattr(block, "text", "") for block in rc
                            )
                        if return_usage:
                            return {
                                "content": output_text,
                                "reasoning": reasoning_text,
                                "usage": usage,
                            }
                        return output_text, reasoning_text

                    if return_usage:
                        return {
                            "content": output_text,
                            "usage": usage,
                        }
                    return output_text

                else:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=prompt,
                        temperature=temperature,
                        # top_p=top_p,
                        max_tokens=max_completion_tokens,
                    )

                    content = resp.choices[0].message.content or ""
                    usage = _extract_usage(resp, is_gpt5=False)

                    if return_reasoning:
                        if return_usage:
                            return {
                                "content": content,
                                "reasoning": None,
                                "usage": usage,
                            }
                        return content, None

                    if return_usage:
                        return {
                            "content": content,
                            "usage": usage,
                        }
                    return content

            except Exception as e:
                if attempt == max_retry:
                    continue
                print(f"[{model_name}] Error on attempt {attempt}: {e}")
                await asyncio.sleep(backoff * (2 ** (attempt - 1)) + random.random())


async def call_with_index(index: int, **kwargs):
    """Keep batch outputs aligned with their original prompt indices."""
    try:
        result = await get_completion_async(**kwargs)
        return index, result
    except Exception:
        return index, None


async def api_call_async(
        model_name: str, 
        user_prompt_list, 
        api_list: list[str], 
        base_url_list: list[str],
        api_call_limit: int,
        max_retry: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_completion_tokens: int = 512,
        return_usage: bool = False):
    """Fan out prompts across one or more OpenAI-compatible API clients."""

    deepseek_model_names = ["deepseek-chat", "deepseek-reasoner"]
    openai_model_names = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-5-mini", "gemini-2.5-pro", "gpt-5.1", "gpt-5"]
    # supported_model_names = deepseek_model_names + openai_model_names
    # if model_name not in supported_model_names:
    #     raise ValueError(f"{model_name} must be of the following names: {', '.join(supported_model_names)}")
    return_reasoning = model_name == "deepseek-reasoner"
    
    
    # Build one client parameter set per API key / endpoint pair.
    client_params_list = []
    if model_name in deepseek_model_names:
        for api in api_list:
            client_params_list.append({
                "api_key": api,
                "base_url": "https://api.deepseek.com",
            })
    else:
        for api, url in zip(api_list, base_url_list):
            client_params_list.append({
                "api_key": api,
                "base_url": url,
            })

    semaphore_list = [asyncio.Semaphore(api_call_limit) for _ in api_list]

    client_prompts = [[] for _ in api_list]
    for i, user_prompt in enumerate(user_prompt_list):
        client_index = i % len(api_list)
        client_prompts[client_index].append((i, user_prompt))


    async def manage_client_tasks(
        client_params: dict, 
        prompts_with_index: list, 
        semaphore: asyncio.Semaphore
    ):
        async with AsyncOpenAI(**client_params) as client:
            tasks = []
            for index, prompt in prompts_with_index:
                task = asyncio.create_task(call_with_index(
                    index=index,
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    return_reasoning=return_reasoning,
                    return_usage=return_usage,
                    max_retry=max_retry,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                    semaphore=semaphore,
                ))
                tasks.append(task)

            if tasks:
                return await asyncio.gather(*tasks)
            return []

    manager_tasks = []
    for params, prompts, semaphore in zip(client_params_list, client_prompts, semaphore_list):
        if prompts:
            manager_task = asyncio.create_task(
                manage_client_tasks(params, prompts, semaphore)
            )
            manager_tasks.append(manager_task)

    all_results_nested = await asyncio.gather(*manager_tasks)

    final_results_with_index = []
    for client_batch_results in all_results_nested:
        final_results_with_index.extend(client_batch_results)

    results = [None] * len(user_prompt_list)
    for index, result in final_results_with_index:
        results[index] = result

    return results


def api_call(
        model_name: str, 
        user_prompt_list, 
        api_list: list[str], 
        base_url_list: list[str],
        api_call_limit: int,
        max_retry: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_completion_tokens: int = 512,
        return_usage: bool = False):
    """Synchronous wrapper around the async batch API caller."""
    
    return asyncio.run(api_call_async(model_name=model_name,  
                                      user_prompt_list=user_prompt_list, 
                                      api_list=api_list, 
                                      base_url_list=base_url_list,
                                      api_call_limit=api_call_limit, 
                                      max_retry=max_retry, 
                                      temperature=temperature, 
                                      top_p=top_p, 
                                      max_completion_tokens=max_completion_tokens,
                                      return_usage=return_usage))
