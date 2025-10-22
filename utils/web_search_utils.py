"""
Web Search Utility Functions

웹 검색, 스크래핑, 뉴스 수집 관련 함수들
"""

import os
import time
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from .logging_utils import log_agent_activity, log_error

# 환경 변수 로드
load_dotenv()

# API 설정
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))


def search_web(
    query: str,
    num_results: int = 10,
    search_type: str = "google"
) -> List[Dict[str, Any]]:
    """
    웹 검색 수행
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수
        search_type: 검색 엔진 타입 ("google", "bing")
        
    Returns:
        검색 결과 리스트 [{"title": "...", "url": "...", "snippet": "..."}, ...]
        
    Examples:
        >>> results = search_web("AI ethics guidelines", num_results=5)
        >>> for result in results:
        ...     print(result["title"])
    """
    if not ENABLE_WEB_SEARCH:
        log_agent_activity(
            agent_name="WebSearch",
            action="search_disabled",
            data={"query": query}
        )
        return []
    
    if not SERPAPI_KEY:
        log_agent_activity(
            agent_name="WebSearch",
            action="no_api_key",
            data={"query": query}
        )
        return _fallback_web_search(query, num_results)
    
    try:
        # SerpAPI를 사용한 Google 검색
        from serpapi import GoogleSearch
        
        params = {
            "q": query,
            "num": num_results,
            "api_key": SERPAPI_KEY,
            "engine": "google"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # 결과 파싱
        parsed_results = []
        
        if "organic_results" in results:
            for result in results["organic_results"][:num_results]:
                parsed_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": "google"
                })
        
        log_agent_activity(
            agent_name="WebSearch",
            action="search_completed",
            data={
                "query": query,
                "num_results": len(parsed_results),
                "search_type": search_type
            }
        )
        
        return parsed_results
    
    except Exception as e:
        log_error("WebSearch", e, {"query": query})
        return _fallback_web_search(query, num_results)


def _fallback_web_search(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    API 없이 기본 웹 검색 (제한적)
    
    Args:
        query: 검색 쿼리
        num_results: 결과 수
        
    Returns:
        검색 결과 리스트
    """
    log_agent_activity(
        agent_name="WebSearch",
        action="using_fallback",
        data={"query": query}
    )
    
    # DuckDuckGo HTML 검색 (API 키 불필요)
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for result_div in soup.select('.result')[:num_results]:
            title_elem = result_div.select_one('.result__title')
            snippet_elem = result_div.select_one('.result__snippet')
            url_elem = result_div.select_one('.result__url')
            
            if title_elem and url_elem:
                results.append({
                    "title": title_elem.get_text(strip=True),
                    "url": url_elem.get('href', ''),
                    "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                    "source": "duckduckgo"
                })
        
        return results
    
    except Exception as e:
        log_error("WebSearch", e, {"query": query, "method": "fallback"})
        return []


def scrape_webpage(url: str, max_length: int = 10000) -> str:
    """
    웹페이지 내용 추출
    
    Args:
        url: 웹페이지 URL
        max_length: 최대 텍스트 길이
        
    Returns:
        추출된 텍스트
        
    Examples:
        >>> text = scrape_webpage("https://example.com/article")
        >>> print(f"Extracted {len(text)} characters")
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 불필요한 요소 제거
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
            tag.decompose()
        
        # 본문 텍스트 추출
        # article, main, content 등의 주요 컨텐츠 영역 우선
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find(class_=['content', 'post', 'article']) or
            soup.body
        )
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # 연속된 줄바꿈 정리
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 길이 제한
        text = text[:max_length]
        
        log_agent_activity(
            agent_name="WebScraper",
            action="page_scraped",
            data={
                "url": url,
                "text_length": len(text)
            }
        )
        
        return text
    
    except Exception as e:
        log_error("WebScraper", e, {"url": url})
        return ""


def search_news(
    keywords: List[str],
    max_results: int = 10,
    language: str = "en"
) -> List[Dict[str, Any]]:
    """
    AI 윤리 관련 최신 뉴스 검색
    
    Args:
        keywords: 검색 키워드 리스트
        max_results: 최대 결과 수
        language: 언어 코드 ("en", "ko" 등)
        
    Returns:
        뉴스 리스트 [{"title": "...", "description": "...", "url": "...", "date": "..."}, ...]
        
    Examples:
        >>> news = search_news(["AI ethics", "bias"], max_results=5)
        >>> for article in news:
        ...     print(article["title"])
    """
    if not NEWS_API_KEY:
        log_agent_activity(
            agent_name="NewsSearch",
            action="no_api_key",
            data={"keywords": keywords}
        )
        return _fallback_news_search(keywords, max_results)
    
    try:
        # NewsAPI 사용
        query = " OR ".join(keywords)
        url = "https://newsapi.org/v2/everything"
        
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": max_results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        news_articles = []
        if data.get("status") == "ok" and "articles" in data:
            for article in data["articles"]:
                news_articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "date": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "")
                })
        
        log_agent_activity(
            agent_name="NewsSearch",
            action="news_retrieved",
            data={
                "keywords": keywords,
                "num_articles": len(news_articles)
            }
        )
        
        return news_articles
    
    except Exception as e:
        log_error("NewsSearch", e, {"keywords": keywords})
        return _fallback_news_search(keywords, max_results)


def _fallback_news_search(keywords: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
    """
    뉴스 검색 대체 방법 (RSS 피드 사용)
    
    Args:
        keywords: 검색 키워드
        max_results: 최대 결과 수
        
    Returns:
        뉴스 리스트
    """
    try:
        import feedparser
        
        # Google News RSS 피드
        query = "+".join(keywords)
        feed_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(feed_url)
        
        news_articles = []
        for entry in feed.entries[:max_results]:
            news_articles.append({
                "title": entry.get("title", ""),
                "description": entry.get("summary", ""),
                "url": entry.get("link", ""),
                "date": entry.get("published", ""),
                "source": "Google News"
            })
        
        log_agent_activity(
            agent_name="NewsSearch",
            action="fallback_news_retrieved",
            data={"num_articles": len(news_articles)}
        )
        
        return news_articles
    
    except Exception as e:
        log_error("NewsSearch", e, {"keywords": keywords, "method": "fallback"})
        return []


def search_academic_papers(
    query: str,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    학술 논문 검색 (arXiv)
    
    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        
    Returns:
        논문 리스트
        
    Examples:
        >>> papers = search_academic_papers("AI fairness", max_results=3)
        >>> for paper in papers:
        ...     print(paper["title"])
    """
    try:
        import arxiv
        
        # arXiv 검색
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            papers.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "url": result.entry_id,
                "published": result.published.isoformat(),
                "categories": result.categories
            })
        
        log_agent_activity(
            agent_name="AcademicSearch",
            action="papers_retrieved",
            data={"query": query, "num_papers": len(papers)}
        )
        
        return papers
    
    except Exception as e:
        log_error("AcademicSearch", e, {"query": query})
        return []


def batch_scrape_urls(urls: List[str], delay: float = 1.0) -> Dict[str, str]:
    """
    여러 URL을 순차적으로 스크래핑
    
    Args:
        urls: URL 리스트
        delay: 요청 사이 지연 시간 (초)
        
    Returns:
        {url: 텍스트} 딕셔너리
        
    Examples:
        >>> urls = ["https://example.com/1", "https://example.com/2"]
        >>> results = batch_scrape_urls(urls)
        >>> print(f"Scraped {len(results)} pages")
    """
    results = {}
    
    for url in urls:
        text = scrape_webpage(url)
        results[url] = text
        
        # Rate limiting을 위한 지연
        if delay > 0:
            time.sleep(delay)
    
    log_agent_activity(
        agent_name="WebScraper",
        action="batch_scrape_completed",
        data={"num_urls": len(urls), "successful": sum(1 for v in results.values() if v)}
    )
    
    return results


def extract_links_from_page(url: str, filter_domain: Optional[str] = None) -> List[str]:
    """
    웹페이지에서 링크 추출
    
    Args:
        url: 웹페이지 URL
        filter_domain: 특정 도메인의 링크만 추출 (선택사항)
        
    Returns:
        URL 리스트
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # 상대 경로를 절대 경로로 변환
            if href.startswith('/'):
                from urllib.parse import urljoin
                href = urljoin(url, href)
            
            # 도메인 필터링
            if filter_domain:
                if filter_domain in href:
                    links.append(href)
            else:
                if href.startswith('http'):
                    links.append(href)
        
        # 중복 제거
        links = list(set(links))
        
        log_agent_activity(
            agent_name="LinkExtractor",
            action="links_extracted",
            data={"url": url, "num_links": len(links)}
        )
        
        return links
    
    except Exception as e:
        log_error("LinkExtractor", e, {"url": url})
        return []


# 테스트 코드
if __name__ == "__main__":
    print("Testing Web Search Utils...")
    
    # 웹 검색 테스트
    print("\n1. Web Search Test:")
    results = search_web("AI ethics", num_results=3)
    print(f"   Found {len(results)} results")
    if results:
        print(f"   First result: {results[0]['title']}")
    
    # 뉴스 검색 테스트
    print("\n2. News Search Test:")
    news = search_news(["AI", "ethics"], max_results=3)
    print(f"   Found {len(news)} news articles")
    if news:
        print(f"   Latest: {news[0]['title']}")
    
    # 웹 스크래핑 테스트
    print("\n3. Web Scraping Test:")
    if results:
        text = scrape_webpage(results[0]['url'])
        print(f"   Scraped {len(text)} characters")
    
    print("\n✓ All tests completed")

