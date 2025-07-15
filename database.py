import json
import os

DB_FILE = 'database.json'

def get_data():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_data(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def add_member(name, age, gender, face_encoding):
    data = get_data()
    data[name] = {
        "age": age,
        "gender": gender,
        "face_encoding": face_encoding.tolist()
    }
    save_data(data)

def is_member(name):
    return name in get_data()

def get_member_encoding(name):
    data = get_data()
    return data.get(name, {}).get("face_encoding")

def get_all_members():
    return get_data()

def clear_database():
    save_data({})