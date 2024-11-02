from egpt import *


apiKey = 'sk-proj-nuTuucC32UGrdZQPGI97T3BlbkFJu1x1SmbrAhneM5IEMZgh'
model = egpt(apiKey=apiKey)

userContext = 'I feel overwhelmed with everything going on in my life.'
predictedResponse = model.respond(userContext)
print('Predicted Response for egpt:', predictedResponse)