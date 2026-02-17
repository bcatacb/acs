#!/usr/bin/env python3
"""
FlowState Backend API Testing Suite
Tests all authentication, project, and genre endpoints
"""

import requests
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

class FlowStateAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token = None
        self.user_id = None
        self.test_user_email = f"test_user_{datetime.now().strftime('%H%M%S')}@example.com"
        self.test_username = f"testuser_{datetime.now().strftime('%H%M%S')}"
        self.test_password = "TestPass123!"
        self.created_project_id = None
        
        # Test results tracking
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        self.passed_tests = []

    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            self.passed_tests.append(name)
            print(f"✅ {name} - PASSED")
        else:
            self.failed_tests.append({"test": name, "details": details})
            print(f"❌ {name} - FAILED: {details}")

    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                    expected_status: int = 200, files: Optional[Dict] = None) -> tuple[bool, Dict]:
        """Make HTTP request and validate response"""
        url = f"{self.base_url}/api/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        if files:
            # Remove Content-Type for file uploads
            headers.pop('Content-Type', None)

        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, headers=headers, files=files, data=data)
                else:
                    response = requests.post(url, headers=headers, json=data)
            elif method == 'PATCH':
                response = requests.patch(url, headers=headers, json=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)
            else:
                return False, {"error": f"Unsupported method: {method}"}

            success = response.status_code == expected_status
            
            try:
                response_data = response.json()
            except:
                response_data = {"status_code": response.status_code, "text": response.text}

            if not success:
                print(f"   Expected status {expected_status}, got {response.status_code}")
                print(f"   Response: {response_data}")

            return success, response_data

        except Exception as e:
            print(f"   Request failed with exception: {str(e)}")
            return False, {"error": str(e)}

    def test_health_check(self):
        """Test basic health endpoints"""
        print("\n🔍 Testing Health Endpoints...")
        
        # Test root endpoint
        success, data = self.make_request('GET', '', expected_status=200)
        self.log_test("Root endpoint (/api/)", success, 
                     "" if success else f"Response: {data}")
        
        # Test health endpoint
        success, data = self.make_request('GET', 'health', expected_status=200)
        self.log_test("Health check (/api/health)", success,
                     "" if success else f"Response: {data}")

    def test_user_registration(self):
        """Test user registration"""
        print("\n🔍 Testing User Registration...")
        
        user_data = {
            "email": self.test_user_email,
            "password": self.test_password,
            "username": self.test_username
        }
        
        success, data = self.make_request('POST', 'auth/register', user_data, expected_status=200)
        
        if success and 'access_token' in data and 'user' in data:
            self.token = data['access_token']
            self.user_id = data['user']['id']
            self.log_test("User registration", True)
        else:
            self.log_test("User registration", False, 
                         f"Missing token or user data: {data}")

    def test_user_login(self):
        """Test user login"""
        print("\n🔍 Testing User Login...")
        
        login_data = {
            "email": self.test_user_email,
            "password": self.test_password
        }
        
        success, data = self.make_request('POST', 'auth/login', login_data, expected_status=200)
        
        if success and 'access_token' in data and 'user' in data:
            # Update token from login
            self.token = data['access_token']
            self.log_test("User login", True)
        else:
            self.log_test("User login", False,
                         f"Missing token or user data: {data}")

    def test_get_current_user(self):
        """Test get current user endpoint"""
        print("\n🔍 Testing Get Current User...")
        
        if not self.token:
            self.log_test("Get current user", False, "No authentication token available")
            return
        
        success, data = self.make_request('GET', 'auth/me', expected_status=200)
        
        if success and 'id' in data and 'email' in data and 'username' in data:
            self.log_test("Get current user", True)
        else:
            self.log_test("Get current user", False,
                         f"Missing user data fields: {data}")

    def test_get_genres(self):
        """Test get genres endpoint"""
        print("\n🔍 Testing Get Genres...")
        
        success, data = self.make_request('GET', 'genres', expected_status=200)
        
        if success and 'genres' in data and isinstance(data['genres'], list) and len(data['genres']) > 0:
            # Check if genres have required fields
            first_genre = data['genres'][0]
            if 'id' in first_genre and 'name' in first_genre and 'description' in first_genre:
                self.log_test("Get genres", True)
            else:
                self.log_test("Get genres", False,
                             f"Genre missing required fields: {first_genre}")
        else:
            self.log_test("Get genres", False,
                         f"Invalid genres response: {data}")

    def test_create_project(self):
        """Test create project endpoint"""
        print("\n🔍 Testing Create Project...")
        
        if not self.token:
            self.log_test("Create project", False, "No authentication token available")
            return
        
        project_data = {
            "name": "Test Project",
            "genre": "trap"
        }
        
        success, data = self.make_request('POST', 'projects', project_data, expected_status=200)
        
        if success and 'id' in data and 'name' in data and 'genre' in data:
            self.created_project_id = data['id']
            self.log_test("Create project", True)
        else:
            self.log_test("Create project", False,
                         f"Missing project data fields: {data}")

    def test_get_projects(self):
        """Test get user projects endpoint"""
        print("\n🔍 Testing Get User Projects...")
        
        if not self.token:
            self.log_test("Get projects", False, "No authentication token available")
            return
        
        success, data = self.make_request('GET', 'projects', expected_status=200)
        
        if success and isinstance(data, list):
            if len(data) > 0:
                # Check if project has required fields
                project = data[0]
                if 'id' in project and 'name' in project and 'genre' in project:
                    self.log_test("Get projects", True)
                else:
                    self.log_test("Get projects", False,
                                 f"Project missing required fields: {project}")
            else:
                self.log_test("Get projects", True)  # Empty list is valid
        else:
            self.log_test("Get projects", False,
                         f"Invalid projects response: {data}")

    def test_get_single_project(self):
        """Test get single project endpoint"""
        print("\n🔍 Testing Get Single Project...")
        
        if not self.token or not self.created_project_id:
            self.log_test("Get single project", False, 
                         "No authentication token or project ID available")
            return
        
        success, data = self.make_request('GET', f'projects/{self.created_project_id}', 
                                        expected_status=200)
        
        if success and 'id' in data and 'name' in data and 'genre' in data:
            self.log_test("Get single project", True)
        else:
            self.log_test("Get single project", False,
                         f"Missing project data fields: {data}")

    def test_update_project(self):
        """Test update project endpoint"""
        print("\n🔍 Testing Update Project...")
        
        if not self.token or not self.created_project_id:
            self.log_test("Update project", False,
                         "No authentication token or project ID available")
            return
        
        update_data = {
            "genre": "boom_bap"
        }
        
        success, data = self.make_request('PATCH', f'projects/{self.created_project_id}',
                                        update_data, expected_status=200)
        
        if success and 'id' in data and data.get('genre') == 'boom_bap':
            self.log_test("Update project", True)
        else:
            self.log_test("Update project", False,
                         f"Project not updated correctly: {data}")

    def test_delete_project(self):
        """Test delete project endpoint"""
        print("\n🔍 Testing Delete Project...")
        
        if not self.token or not self.created_project_id:
            self.log_test("Delete project", False,
                         "No authentication token or project ID available")
            return
        
        success, data = self.make_request('DELETE', f'projects/{self.created_project_id}',
                                        expected_status=200)
        
        if success and 'message' in data:
            self.log_test("Delete project", True)
            self.created_project_id = None  # Clear the ID since it's deleted
        else:
            self.log_test("Delete project", False,
                         f"Unexpected delete response: {data}")

    def test_unauthorized_access(self):
        """Test endpoints without authentication"""
        print("\n🔍 Testing Unauthorized Access...")
        
        # Temporarily clear token
        original_token = self.token
        self.token = None
        
        # Test protected endpoint without auth
        success, data = self.make_request('GET', 'auth/me', expected_status=401)
        self.log_test("Unauthorized access to /auth/me", success,
                     "" if success else f"Should return 401, got: {data}")
        
        success, data = self.make_request('GET', 'projects', expected_status=401)
        self.log_test("Unauthorized access to /projects", success,
                     "" if success else f"Should return 401, got: {data}")
        
        # Restore token
        self.token = original_token

    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting FlowState Backend API Tests")
        print(f"📍 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Run test suites in order
        self.test_health_check()
        self.test_user_registration()
        self.test_user_login()
        self.test_get_current_user()
        self.test_get_genres()
        self.test_create_project()
        self.test_get_projects()
        self.test_get_single_project()
        self.test_update_project()
        self.test_delete_project()
        self.test_unauthorized_access()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {len(self.failed_tests)}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  • {test['test']}: {test['details']}")
        
        if self.passed_tests:
            print(f"\n✅ PASSED TESTS ({len(self.passed_tests)}):")
            for test in self.passed_tests:
                print(f"  • {test}")
        
        return len(self.failed_tests) == 0

def main():
    """Main test runner"""
    tester = FlowStateAPITester()
    success = tester.run_all_tests()
    
    # Return appropriate exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
