from pydantic import BaseModel

'''
Pydantic - is used to define data models just like zod for javascript/typescript.

    - Pydantic also try to convert string to valid datatype like to integer or boolean, but if fails then raise an error.
'''
class User(BaseModel):
    id: int
    name: str
    is_active: bool

input_data = {
    "id": 1,
    "name": "code",
    "is_active": True,
}

user = User(**input_data)
print(user)