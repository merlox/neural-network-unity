using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour {
    private ArtificialNeuralNetwork artificialNeuralNetwork;
    private double sumSquareError = 0;
    public int runTimes = 1000;
    public double trainingRate = 0.8;

    private void Start()
    {
        // int amountInputs, int amountOutputs, int amountHiddenLayers, int amountNeuronsPerHiddenLayer, double alpha
        artificialNeuralNetwork = new ArtificialNeuralNetwork(2, 1, 1, 2, trainingRate);
        List<double> result;

        if(runTimes == 1)
        {
            sumSquareError = 0;
            result = Train(1, 1, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
        }
        else
        {
            // Train it 1000 times with those 4 cases XOR
            for (int i = 0; i < runTimes; i++)
            {
                sumSquareError = 0;
                result = Train(1, 1, 0);
                sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
                result = Train(1, 0, 1);
                sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
                result = Train(0, 1, 1);
                sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
                result = Train(0, 0, 0);
                sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            }
        }

        Debug.Log("Sum squared errors: " + sumSquareError);
        result = Train(1, 1, 0);
        Debug.Log("1 1 : " + result[0]);
        result = Train(1, 0, 1);
        Debug.Log("1 0 : " + result[0]);
        result = Train(0, 1, 1);
        Debug.Log("0 1 : " + result[0]);
        result = Train(0, 0, 0);
        Debug.Log("0 0 : " + result[0]);
    }

    private List<double> Train(double firstInput, double secondInput, double output)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(firstInput);
        inputs.Add(secondInput);
        outputs.Add(output);
        List<double> result = artificialNeuralNetwork.ExecuteNeuralNetwork(inputs, outputs);
        return result;
    }
}
