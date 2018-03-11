using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer {
    public int amountNeurons;
    public List<Neuron> neurons = new List<Neuron>();

    public Layer(int numberNeurons, int numberNeuronInputs)
    {
        amountNeurons = numberNeurons;
        for (int i = 0; i < numberNeurons; i++) neurons.Add(new Neuron(numberNeuronInputs));
    }
}
