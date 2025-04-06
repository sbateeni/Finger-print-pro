# API Documentation

This document provides detailed information about the Fingerprint Recognition System's API endpoints and their usage.

## Base URL

```
http://localhost:5000/api/v1
```

## Authentication

All API endpoints require authentication using a JWT token. Include the token in the Authorization header:

```
Authorization: Bearer <your-token>
```

## Endpoints

### Authentication

#### POST /auth/register
Register a new user.

**Request Body:**
```json
{
    "username": "string",
    "password": "string",
    "email": "string"
}
```

**Response:**
```json
{
    "message": "User registered successfully",
    "user_id": "string"
}
```

#### POST /auth/login
Login and receive a JWT token.

**Request Body:**
```json
{
    "username": "string",
    "password": "string"
}
```

**Response:**
```json
{
    "access_token": "string",
    "token_type": "bearer"
}
```

### Fingerprint Operations

#### POST /fingerprints/upload
Upload a new fingerprint image.

**Request Body:**
- `file`: Fingerprint image file (JPEG, PNG)
- `metadata`: JSON string containing additional information

**Response:**
```json
{
    "fingerprint_id": "string",
    "status": "success",
    "message": "Fingerprint uploaded successfully"
}
```

#### GET /fingerprints/{fingerprint_id}
Retrieve a fingerprint by ID.

**Response:**
```json
{
    "fingerprint_id": "string",
    "image_url": "string",
    "metadata": {
        "upload_date": "string",
        "quality_score": "float",
        "features": {
            "minutiae_count": "integer",
            "core_point": {
                "x": "integer",
                "y": "integer"
            },
            "delta_point": {
                "x": "integer",
                "y": "integer"
            }
        }
    }
}
```

#### POST /fingerprints/match
Match a fingerprint against the database.

**Request Body:**
- `file`: Fingerprint image file (JPEG, PNG)
- `threshold`: Optional float (0.0 to 1.0) for matching confidence

**Response:**
```json
{
    "matches": [
        {
            "fingerprint_id": "string",
            "confidence": "float",
            "metadata": {
                "upload_date": "string",
                "quality_score": "float"
            }
        }
    ],
    "processing_time": "float"
}
```

### User Management

#### GET /users/profile
Get the current user's profile.

**Response:**
```json
{
    "username": "string",
    "email": "string",
    "registration_date": "string",
    "fingerprint_count": "integer"
}
```

#### PUT /users/profile
Update user profile.

**Request Body:**
```json
{
    "email": "string",
    "current_password": "string",
    "new_password": "string"
}
```

**Response:**
```json
{
    "message": "Profile updated successfully"
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
    "error": "Invalid request",
    "message": "Detailed error message"
}
```

### 401 Unauthorized
```json
{
    "error": "Unauthorized",
    "message": "Invalid or expired token"
}
```

### 404 Not Found
```json
{
    "error": "Not found",
    "message": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
    "error": "Server error",
    "message": "Internal server error occurred"
}
```

## Rate Limiting

API requests are limited to:
- 100 requests per minute for authenticated users
- 20 requests per minute for unauthenticated users

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625097600
```

## Pagination

Endpoints that return lists support pagination using query parameters:
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 10, max: 100)

Pagination metadata is included in the response:
```json
{
    "items": [...],
    "total": 100,
    "page": 1,
    "per_page": 10,
    "total_pages": 10
}
``` 