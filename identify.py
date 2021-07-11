import face_recognition as fr 
from PIL import Image, ImageDraw

img_yash = fr.load_image_file(r'Face_Recognition\Known\Yash Dhingra.jpg')
encode_yash = fr.face_encodings(img_yash)[0]

img_yash2 = fr.load_image_file(r'Face_Recognition\Known\Yash Dhingra2.jpg')
encode_yash2 = fr.face_encodings(img_yash2)[0]

img_yash3 = fr.load_image_file(r'Face_Recognition\Known\Yash Dhingra3.png')
encode_yash3 = fr.face_encodings(img_yash3)[0]

img_pooja = fr.load_image_file(r'Face_Recognition\Known\Pooja Dhingra.jpg')
encode_pooja = fr.face_encodings(img_pooja)[0]

img_pooja2 = fr.load_image_file(r'Face_Recognition\Known\Pooja Dhingra2.jpg')
encode_pooja2 = fr.face_encodings(img_pooja2)[0]


img_aarav = fr.load_image_file(r'Face_Recognition\Known\Aarav Dhingra.jpg')
encode_aarav = fr.face_encodings(img_aarav)[0]

img_rajat = fr.load_image_file(r'Face_Recognition\Known\Rajat Dhingra.jpg')
encode_rajat = fr.face_encodings(img_rajat)[0]

known_face_encodings = [
    encode_yash,
    encode_yash2,
    encode_yash3,
    encode_pooja,
    encode_pooja2,
    encode_aarav,
    encode_rajat
]

known_face_names=[
    'Yash Dhingra1',
    'Yash Dhingra2',
    'Yash Dhingra3',
    'Pooja Dhingra',
    'Pooja Dhingra2',
    'Aarav Dhingra',
    'Rajat Dhingra'
]

img_tmp = fr.load_image_file(r'Face_Recognition\Groups\IMG_1326.jpg')
face_locations = fr.face_locations(img_tmp)
encode_img = fr.face_encodings(img_tmp,face_locations)

pil_img = Image.fromarray(img_tmp)

draw = ImageDraw.Draw(pil_img)

for (top, right, bottom, left), face_encoding in zip(face_locations,encode_img):
    matches = fr.compare_faces(known_face_encodings,face_encoding)
    name = "Unknown Face Detected"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    draw.rectangle(((left,top),(right,bottom)),outline=(0,0,0))
    text_width,text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 10),(right,bottom)),fill=(0,0,0),outline=(0,0,0))
    draw.text((left - 6, bottom - text_height - 5),name,fill=(255,255,255))
del draw
pil_img.show()