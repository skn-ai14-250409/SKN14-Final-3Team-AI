#!/usr/bin/env python3
"""
KB RAG 시스템 데이터 관리 도구
- 데이터 업로드
- 벡터 스토어 관리  
- 시스템 상태 확인
"""
import requests
import time
import json
from pathlib import Path
import argparse

BASE_URL = "http://localhost:8000/api/v1"

class DataManager:
    def __init__(self):
        self.base_url = BASE_URL
    
    def check_server(self):
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/healthcheck", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_vector_stats(self):
        """벡터 스토어 통계 조회"""
        try:
            response = requests.get(f"{self.base_url}/vector_store_stats")
            if response.status_code == 200:
                return response.json().get("stats", {})
        except:
            pass
        return {}
    
    def upload_all_data(self):
        """전체 데이터 업로드"""
        print("전체 데이터 업로드 시작...")
        
        try:
            response = requests.post(
                f"{self.base_url}/initialize_vector_store",
                timeout=1800  # 30분으로 증가
            )
            
            if response.status_code == 200:
                data = response.json()
                print("업로드 완료!")
                print(f"총 문서: {data.get('documents_count', 0)}개")
                print(f"청크: {len(data.get('chunk_ids', []))}개")
                return True
            else:
                print(f"업로드 실패: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print("타임아웃: 업로드가 오래 걸리고 있습니다.")
            return False
        except Exception as e:
            print(f"오류: {e}")
            return False
    
    def upload_folder(self, folder_name):
        """특정 폴더 업로드"""
        print(f"폴더 업로드: {folder_name}")
        
        try:
            response = requests.post(
                f"{self.base_url}/ingest_folder",
                json={"root_folder_path": folder_name},
                timeout=1200  # 20분으로 증가
            )
            
            if response.status_code == 200:
                data = response.json()
                chunk_count = data.get("uploaded_chunk_ids_count", 0)
                print(f"성공: {chunk_count}개 청크 업로드")
                return True
            else:
                print(f"실패: {response.text}")
                return False
                
        except Exception as e:
            print(f"오류: {e}")
            return False
    
    def upload_file(self, file_path):
        """개별 파일 업로드"""
        print(f"파일 업로드: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없습니다: {file_path}")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                files = {'files': (os.path.basename(file_path), f, 'application/pdf')}
                response = requests.post(
                    f"{self.base_url}/upload_docs_to_rag",
                    files=files,
                    timeout=300  # 5분
                )
            
            if response.status_code == 200:
                data = response.json()
                chunk_count = len(data.get("chunk_ids", []))
                print(f"성공: {chunk_count}개 청크 업로드")
                return True
            else:
                print(f"실패: {response.text}")
                return False
                
        except Exception as e:
            print(f"오류: {e}")
            return False
    
    def delete_all_vectors(self):
        """모든 벡터 삭제"""
        print("모든 벡터 삭제 중...")
        
        try:
            response = requests.delete(f"{self.base_url}/delete_all_vectors")
            if response.status_code == 200:
                print("모든 벡터가 삭제되었습니다.")
                return True
            else:
                print(f"삭제 실패: {response.status_code}")
                return False
        except Exception as e:
            print(f"오류: {e}")
            return False
    
    def delete_vectors_by_condition(self, field: str, value: str):
        """특정 조건에 맞는 벡터만 삭제"""
        print(f"조건 삭제: {field} = {value}")
        
        try:
            response = requests.delete(
                f"{self.base_url}/delete_vectors_by_condition",
                json={"field": field, "value": value}
            )
            
            if response.status_code == 200:
                data = response.json()
                deleted_count = data.get("deleted_count", 0)
                print(f"삭제 완료: {deleted_count}개 벡터 삭제됨")
                return True
            else:
                print(f"삭제 실패: {response.text}")
                return False
                
        except Exception as e:
            print(f"오류: {e}")
            return False
    
    def show_status(self):
        """시스템 상태 출력"""
        print("시스템 상태")
        print("=" * 40)
        
        # 서버 상태
        if self.check_server():
            print("서버: 정상")
        else:
            print("서버: 연결 불가")
            return
        
        # 벡터 스토어 통계
        stats = self.get_vector_stats()
        if stats:
            print(f"벡터 개수: {stats.get('total_vectors', 0):,}")
            print(f"차원: {stats.get('dimension', 0)}")
            print(f"인덱스 사용률: {stats.get('index_fullness', 0)*100:.1f}%")
        else:
            print("벡터 스토어: 정보 없음")

def main():
    parser = argparse.ArgumentParser(description="KB RAG 데이터 관리 도구")
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 상태 확인
    subparsers.add_parser('status', help='시스템 상태 확인')
    
    # 데이터 업로드
    upload_parser = subparsers.add_parser('upload', help='데이터 업로드')
    upload_parser.add_argument('--folder', help='특정 폴더만 업로드 (예: 상품)')
    upload_parser.add_argument('--file', help='개별 파일 업로드 (예: 강령/공통/윤리강령.pdf)')
    upload_parser.add_argument('--all', action='store_true', help='전체 데이터 업로드')
    
    # 벡터 삭제
    subparsers.add_parser('clear', help='모든 벡터 삭제')
    
    # 조건부 벡터 삭제
    delete_parser = subparsers.add_parser('delete', help='특정 조건에 맞는 벡터 삭제')
    delete_parser.add_argument('--field', required=True, help='삭제할 필드명 (예: file_name, main_category)')
    delete_parser.add_argument('--value', required=True, help='삭제할 값 (예: "경제전망보고서(2025.05).pdf")')
    
    args = parser.parse_args()
    manager = DataManager()
    
    # 서버 연결 확인
    if not manager.check_server():
        print("서버에 연결할 수 없습니다.")
        print("서버를 먼저 실행해주세요: python run_server.py --reload")
        return
    
    if args.command == 'status':
        manager.show_status()
    
    elif args.command == 'upload':
        if args.all:
            manager.upload_all_data()
        elif args.folder:
            manager.upload_folder(args.folder)
        elif args.file:
            manager.upload_file(args.file)
        else:
            print("--all, --folder, 또는 --file 옵션을 지정해주세요.")
    
    elif args.command == 'clear':
        confirm = input("모든 벡터를 삭제하시겠습니까? (y/N): ")
        if confirm.lower() == 'y':
            manager.delete_all_vectors()
        else:
            print("취소되었습니다.")
    
    elif args.command == 'delete':
        confirm = input(f"'{args.field}' = '{args.value}' 조건에 맞는 벡터를 삭제하시겠습니까? (y/N): ")
        if confirm.lower() == 'y':
            manager.delete_vectors_by_condition(args.field, args.value)
        else:
            print("취소되었습니다.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
