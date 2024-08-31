# views.py
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .Crash import run_crash_analysis
# import train_rnn
from .train import predict_next_multiplier
from .train_rnn import predict_rnn

class CrashAPIView(APIView):
    def get(self, request, *args, **kwargs):
        results = predict_next_multiplier()
        # results = train_rnn.predict_rnn() #run_crash_analysis()
        return Response(results)
