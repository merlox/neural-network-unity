using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron {
    // How many inputs you have
    public int amountInputs;
    // The bias
    public double bias;
    // The result output
    public double output;
    // The error
    public double errorGradient;
    // The weights of the inputs
    public List<double> weights = new List<double>();
    // The values of the inputs
    public List<double> inputs = new List<double>();

    // Initialize the weights and bias with random values
    public Neuron(int numberInputs)
    {
        bias = UnityEngine.Random.Range(-1.0f, 1.0f);
        amountInputs = numberInputs;
        for (int i = 0; i < numberInputs; i++) weights.Add(UnityEngine.Random.Range(-1.0f, 1.0f));
    }
}
