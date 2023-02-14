How to run the code:

1. Clone the code the run the train_cnn.ipynb to get the model

2. Install virtualenv : pip install virtualenv

3. Create the virtual environment : virtualenv env

4. Activate it : source env/bin/activate

5. Install dependencies : pip install -r requirements.txt

6. To run the final application, execute : streamlit run deploy_model.py

* It is coded in the deployment that the model will be used once it's properly generated
** To run the model on local: http://localhost:8501 by default
if you'd like to change the port, use the following line

streamlit run my_app.py --server.port (port number)
