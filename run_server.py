#!/usr/bin/env python3
"""
KB금융 RAG API 서버 실행 스크립트

사용법:
    python run_server.py [--host HOST] [--port PORT] [--reload]

예시:
    python run_server.py --host 0.0.0.0 --port 8000 --reload
"""
import argparse
import uvicorn
import logging

def main():
    parser = argparse.ArgumentParser(description="KB금융 RAG API 서버 실행")
    parser.add_argument("--host", default="127.0.0.1", help="서버 호스트 (기본값: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 (기본값: 8000)")
    parser.add_argument("--reload", action="store_true", help="개발 모드 - 파일 변경시 자동 재시작")
    parser.add_argument("--workers", type=int, default=1, help="워커 프로세스 수 (기본값: 1)")
    
    args = parser.parse_args()
    
    # 로깅 설정 (main.py에서 이미 설정됨)
    logger = logging.getLogger(__name__)
    
    logger.info(f"KB금융 RAG API 서버를 시작합니다...")
    logger.info(f"호스트: {args.host}")
    logger.info(f"포트: {args.port}")
    logger.info(f"리로드 모드: {args.reload}")
    
    try:
        uvicorn.run(
            "src.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            access_log=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("서버가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"서버 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
