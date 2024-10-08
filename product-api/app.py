from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# In-memory structure to store products
products = {}

# Function to generate unique IDs for products
def generate_product_id():
    return str(len(products) + 1)

# Route to add a new product
@app.route('/products', methods=['POST'])
def create_product():
    if not request.json or 'name' not in request.json:
        abort(400, description="Product name is required.")
    
    product_id = generate_product_id()
    new_product = {
        'id': product_id,
        'name': request.json['name'],
        'description': request.json.get('description', ''),
        'price': request.json.get('price', 0.0)
    }
    products[product_id] = new_product
    return jsonify(new_product), 201

# Route to retrieve all products
@app.route('/products', methods=['GET'])
def get_products():
    return jsonify(list(products.values()))

# Route to retrieve a product by ID
@app.route('/products/<product_id>', methods=['GET'])
def get_product(product_id):
    product = products.get(product_id)
    if product is None:
        abort(404, description="Product not found.")
    return jsonify(product)

# Route to update an existing product
@app.route('/products/<product_id>', methods=['PUT'])
def update_product(product_id):
    product = products.get(product_id)
    if product is None:
        abort(404, description="Product not found.")
    
    if not request.json:
        abort(400, description="Request body must be JSON.")
    
    product['name'] = request.json.get('name', product['name'])
    product['description'] = request.json.get('description', product['description'])
    product['price'] = request.json.get('price', product['price'])
    
    return jsonify(product)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Product API!"})

# Route to delete a product
@app.route('/products/<product_id>', methods=['DELETE'])
def delete_product(product_id):
    product = products.pop(product_id, None)
    if product is None:
        abort(404, description="Product not found.")
    return jsonify({'result': 'Product deleted'})

# Error handling for bad requests
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad Request', 'message': error.description}), 400

# Error handling for not found resources
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not Found', 'message': error.description}), 404

if __name__ == '__main__':
    app.run(debug=True)
