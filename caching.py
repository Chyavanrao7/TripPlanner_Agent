import os
import json
import redis.asyncio as redis
from functools import wraps
from typing import Callable, Any


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1") # Using DB 1 for cache


try:
    redis_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
    print("✅ Redis connection pool for caching initialized.")
except Exception as e:
    redis_pool = None
    print(f"⚠️ Could not connect to Redis for caching: {e}")

def redis_cache(ttl: int = 3600):
    """
    An asynchronous decorator to cache function results in Redis with a TTL.
    
    Args:
        ttl (int): The time-to-live for the cache key in seconds. Default is 1 hour.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if not redis_pool:
                # If Redis is not available, just call the function directly
                return await func(*args, **kwargs)

            sorted_kwargs = sorted(kwargs.items())
            key_parts = [func.__name__] + [str(a) for a in args] + [f"{k}={v}" for k, v in sorted_kwargs]
            cache_key = f"cache:{':'.join(key_parts)}"

            redis_client = redis.Redis(connection_pool=redis_pool)
            
            try:

                cached_result = await redis_client.get(cache_key)
                
                if cached_result:
                    print(f"✅ Cache HIT for key: {cache_key}")
       
                    return json.loads(cached_result)
                

                print(f"❌ Cache MISS for key: {cache_key}. Calling function...")
                result = await func(*args, **kwargs)
                
                await redis_client.setex(cache_key, ttl, json.dumps(result))
                
                return result
            
            except Exception as e:
                print(f"❌ Caching error for {cache_key}: {e}. Calling function directly.")

                return await func(*args, **kwargs)
            finally:
                await redis_client.close()

        return wrapper
    return decorator
