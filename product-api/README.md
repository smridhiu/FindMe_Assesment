# Product API

## Project Overview

The **Product API** is a RESTful web service built with Flask that allows users to manage a collection of product information. This API supports CRUD (Create, Read, Update, Delete) operations, enabling users to add new products, retrieve product details, update existing products, and delete products from the in-memory data structure.

## Features

- **In-memory data storage** for easy testing and development.
- RESTful endpoints for performing CRUD operations.
- JSON request/response formats with appropriate HTTP status codes.
- Error handling for common scenarios (e.g., resource not found, bad requests).

## Setup Instructions

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**

   Open your terminal and run the following command to clone the repository:

   ```bash
   git clone https://github.com/smridhiu/FindMe_Assesment.git
   cd product-api

2. **Create a Virtual Environment**

It is recommended to create a virtual environment to manage your dependencies:

python -m venv venv
venv\Scripts\activate

3. **Install dependencies**

pip install Flask


4. **Usage Instructions**
To run the application, follow these steps:

Start the Flask Server

Make sure you are in the project directory and the virtual environment is activated, then run:


python app.py

The API will start, and you should see output indicating it is running on http://127.0.0.1:5000.

5. **Testing the Endpoints**

You can use tools like Postman or cURL to test the API endpoints. Below are examples of how to interact with the API.

Endpoint Documentation
1. Add Product
URL: /products
Method: POST
Request Body: (raw JSON)

{
  "name": "Sample Product",
  "price": 19.99
}
Response:
Status: 201 Created
Body:


{
  "id": "1",
  "name": "Sample Product",
  "price": 19.99
}
2. Get All Products
URL: /products
Method: GET
Response:
Status: 200 OK
Body:

[
  {
    "id": "1",
    "name": "Sample Product",
    "price": 19.99
  }
]
3. Get Single Product
URL: /products/<id>
Method: GET
Response:
Status: 200 OK
Body:

{
  "id": "1",
  "name": "Sample Product",
  "price": 19.99
}
Error Response (if not found):
Status: 404 Not Found
Body:

{
  "error": "Not Found",
  "message": "The requested URL was not found on the server."
}
4. Update Product
URL: /products/<id>
Method: PUT
Request Body: (raw JSON)

{
  "name": "Updated Product",
  "price": 29.99
}
Response:
Status: 200 OK
Body:

{
  "id": "1",
  "name": "Updated Product",
  "price": 29.99
}
Error Response (if not found):
Status: 404 Not Found
Body:

{
  "error": "Not Found",
  "message": "The requested product was not found."
}
5. Delete Product
URL: /products/<id>
Method: DELETE
Response:
Status: 200 OK
Body:

{
  "result": true
}
Error Response (if not found):
Status: 404 Not Found
Body:

{
  "error": "Not Found",
  "message": "The requested product was not found."
}
Error Handling
The API includes error handling for various scenarios:

400 Bad Request: Returned when the request data is invalid or missing required fields.
404 Not Found: Returned when trying to access a product that does not exist.