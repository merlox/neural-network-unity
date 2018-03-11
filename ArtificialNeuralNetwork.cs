using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArtificialNeuralNetwork {
    public int amountInputs;
    public int amountOutputs;
    public int amountHiddenLayers;
    public int amountNeuronsPerHiddenLayer;
    // Similar to a weight, a value to determines how fast the weights are modified, the learning rate
    public double alpha;
    private List<Layer> layers = new List<Layer>();

    public ArtificialNeuralNetwork(int amountInputs, int amountOutputs, int amountHiddenLayers, int amountNeuronsPerHiddenLayer, double alpha)
    {
        this.amountInputs = amountInputs;
        this.amountOutputs = amountOutputs;
        this.amountHiddenLayers = amountHiddenLayers;
        this.amountNeuronsPerHiddenLayer = amountNeuronsPerHiddenLayer;
        this.alpha = alpha;

        if(amountHiddenLayers > 0)
        {
            // Create the input layer
            layers.Add(new Layer(amountNeuronsPerHiddenLayer, amountInputs));
            // Create the hidden layers. Note that it only works with the same input and outputs. Note that in the tutorial the condition is: i < amountHiddenLayers - 1;
            for (int i = 0; i < amountHiddenLayers - 1; i++) layers.Add(new Layer(amountNeuronsPerHiddenLayer, amountNeuronsPerHiddenLayer));
            // Create the output layer
            layers.Add(new Layer(amountOutputs, amountNeuronsPerHiddenLayer));
        }
        else
        {
            layers.Add(new Layer(amountOutputs, amountInputs));
        }
    }

    // Returns the outputs of the neural network
    public List<double> ExecuteNeuralNetwork(List<double> inputValues, List<double> desiredOutput)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        // Make sure the amount of inputs is the same or you won't be able to process it
        if(inputValues.Count != amountInputs)
        {
            Debug.Log("Error, the number of inputs must be " + amountInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);
        for(int i = 0; i < amountHiddenLayers + 1; i++)
        {
            // If we are on the next layer after the initial input layer, set the inputs as the outputs of the past layer
            if(i > 0) inputs = new List<double>(outputs);
            outputs.Clear();
            // Loop through all the neurons of the layer to calculate them values
            for(int a = 0; a < layers[i].amountNeurons; a++)
            {
                double sumWeightsInputs = 0;
                layers[i].neurons[a].inputs.Clear();
                // Loop through each neuron input
                for(int b = 0; b < layers[i].neurons[a].amountInputs; b++)
                {
                    // Update each input of each neuron with the new input values, which happen to be the outputs of the last layer if we're on the second or major layer
                    layers[i].neurons[a].inputs.Add(inputs[b]);
                    // Multiply the weights by the value of the input and add it to the sum
                    sumWeightsInputs += layers[i].neurons[a].weights[b] * inputs[b];
                }
                sumWeightsInputs -= layers[i].neurons[a].bias;

                if (i == amountHiddenLayers) layers[i].neurons[a].output = Sigmoid(sumWeightsInputs);
                else layers[i].neurons[a].output = Sigmoid(sumWeightsInputs);

                // After calculating the output activation function of this neuron, add the result output to the outputs array
                outputs.Add(layers[i].neurons[a].output);
            }
        }

        UpdateWeights(outputs, desiredOutput);
        return outputs;
    }

    private void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        // Back propagation, we are going from the end of the neural network to the beginning 
        for(int i = amountHiddenLayers; i >= 0; i--)
        {
            // Loop through all the neurons of this layer
            for (int a = 0; a < layers[i].amountNeurons; a++)
            {
                // If we are in the output layer
                if(i == amountHiddenLayers)
                {
                    // Here's how the error is calculated
                    error = desiredOutput[a] - outputs[a];
                    // Error gradient calculated with the delta rule. The error gradient is the amount of error this neuron contains. All the error gradients combined = total error
                    layers[i].neurons[a].errorGradient = outputs[a] * (1 - outputs[a]) * error;
                }
                else
                {
                    layers[i].neurons[a].errorGradient = layers[i].neurons[a].output * (1 - layers[i].neurons[a].output);
                    double errorGradientSum = 0;
                    for (int b = 0; b < layers[i + 1].amountNeurons; b++)
                    {
                        errorGradientSum += layers[i + 1].neurons[b].errorGradient * layers[i + 1].neurons[b].weights[a];
                    }
                    // This was the error that I was having. You have to multiply the error gradient by the sum or gradients
                    layers[i].neurons[a].errorGradient *= errorGradientSum;
                }

                // Loop through the inputs of this neuron
                for(int c = 0; c < layers[i].neurons[a].amountInputs; c++)
                {
                    // If we are at the last output layer update the weights by comparing the output with the desired output
                    if(i == amountHiddenLayers)
                    {
                        error = desiredOutput[a] - outputs[a];
                        layers[i].neurons[a].weights[c] += alpha * layers[i].neurons[a].inputs[c] * error;
                    }
                    else
                    {
                        layers[i].neurons[a].weights[c] += alpha * layers[i].neurons[a].inputs[c] * layers[i].neurons[a].errorGradient;
                    }
                }
                layers[i].neurons[a].bias += alpha * -1 * layers[i].neurons[a].errorGradient;
            }
        }
    }

    private double ActivationFunction(double value)
    {
        return Sigmoid(value);
    }

    private double ActivationFunctionOutput(double value)
    {
        return Sigmoid(value);
    }

    private double StepActivationFunction(double value)
    {
        if (value < 0) return 0;
        else return 1;
    }

    private double Sigmoid(double value)
    {
        double k = (double)System.Math.Exp(value);
        return k / (1.0f + k);
    }

    private double TanH(double value)
    {
        return (2 * (Sigmoid(2 * value)) - 1);
    }

    private double ReLu(double value)
    {
        if (value > 0) return 0;
        else return 0;
    }

    private double LeakyReLu(double value)
    {
        if (value < 0) return 0.01 * value;
        else return value;
    }
}
