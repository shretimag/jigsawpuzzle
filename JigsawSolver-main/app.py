import streamlit as st 
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image


st.beta_set_page_config(page_title="Puzzle Solver", page_icon="🏙️", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def load_image(img_file):
    img = Image.open(img_file)
    img = np.asarray(img)
    return img

def crop_images(img_arr):
    ret = []
    pieces = []
    for i in range(6):
        for j in range(6):
            pieces.append(img_arr[i*50:(i+1)*50, j*50:(j+1)*50])
    ret.append(pieces)
    return np.array(ret)

def main():

    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAREDN;text-align:left;">Jigsaw Solver</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander("Information", expanded=True):
            st.write("""
            This is a simple website to solve 6x6 puzzles primarily consisting of faces and landmarks. 
            """)



    with col2:
        st.subheader("Upload the puzzle image in 6x6 form")
        
        image_file = st.file_uploader("Upload the puzzle image", type=['png','jpg','jpeg'])
        
        if image_file is not None:
            file_details = {"filename" : image_file.name, "filetype" : image_file.type, "filesize" : image_file.size}
            st.write(file_details)
            st.image(load_image(image_file),width = 300)
        
        if st.button('Solve for Faces'):

            loaded_model = load_model('models/model_faces.pkl')
            img1 = load_image(image_file)
            img = crop_images(img1)
            output = loaded_model.predict(img)
            output = np.argmax(output, axis=-1)
            st.write(output)	
            new_im = np.zeros(img1.shape)
            cut = 50
            for i in range(6):
                for j in range(6):
                    r1 = output[0][i*6 + j]
                    r = r1//6
                    c = r1%6
                    new_im[r*cut:(r+1)*cut, c*cut:(c+1)*cut] = img1[i*cut:(i+1)*cut, j*cut:(j+1)*cut]
            #new_im is the output
            final = Image.fromarray(new_im.astype(np.uint8))

            col1.write('''
		    ## Results 
		    ''')
            col1.success(st.image(final,caption="Solution Image"))

        if st.button('Solve for Landmarks'):

            loaded_model = load_model('models/model_landmarks.pkl')
            img1 = load_image(image_file)
            img = crop_images(img1)
            output = loaded_model.predict(img)
            output = np.argmax(output, axis=-1)
            st.write(output)	
            new_im = np.zeros(img1.shape)
            cut = 50
            for i in range(6):
                for j in range(6):
                    r1 = output[0][i*6 + j]
                    r = r1//6
                    c = r1%6
                    new_im[r*cut:(r+1)*cut, c*cut:(c+1)*cut] = img1[i*cut:(i+1)*cut, j*cut:(j+1)*cut]
            #new_im is the output
            final = Image.fromarray(new_im.astype(np.uint8))

            col1.write('''
		    ## Results 
		    ''')
            col1.success(st.image(final,caption="Solution Image"))
		
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
