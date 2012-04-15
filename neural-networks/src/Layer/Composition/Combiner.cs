using System;
using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public class Combiner<InputT, OutputT> : ISingleLayer<InputT[], OutputT[]> where InputT : IMatrix<InputT>
	{
		ISingleLayer<InputT, OutputT>[] layers;
		OutputT[] outputs;
		InputT[]  errors;
		
		int inputCount;
		int layerCount;
		int outputCount;
		
		public Combiner(ISingleLayer<InputT, OutputT>[] parLayers, int inputCountA)
		{
			layers  = parLayers;
			
			inputCount  = inputCountA;
			layerCount  = layers.Length;
			outputCount = layerCount * inputCount;
				
			outputs = new OutputT[outputCount];
			errors  = new InputT[inputCount];
		}
		
		
		#region ISingleLayer[InputT[],OutputT[]] implementation
		
		public OutputT[] FeedForward (InputT[] input)
		{
			if (input.Length != inputCount)
				throw new ArgumentException("Incorrect count of input layers.");
			
			for (var inputIndex = 0; inputIndex < inputCount; ++inputIndex)
				for (var layerIndex = 0; layerIndex < layerCount; ++layerIndex)
				{
					var outputIndex = inputIndex + layerIndex * inputCount;
					outputs[outputIndex] = layers[layerIndex].FeedForward(input[inputIndex]);
				}
			return outputs;
		}

		public InputT[] PropagateBackward (InputT[] input, OutputT[] outputError)
		{
			if (input.Length * layerCount != outputError.Length || outputError.Length != outputCount)
				throw new ArgumentException("Count of layers do not match.");
			
			for (var inputIndex = 0; inputIndex < inputCount; ++inputIndex)
			{
				errors[inputIndex] = input[inputIndex].ZeroCopy();
				
				for (var layerIndex = 0; layerIndex < layerCount; ++layerIndex)
				{
					var errorIndex = inputIndex + layerIndex * inputCount;
					var error = layers[layerIndex].PropagateBackward(input[inputIndex], outputError[errorIndex]);
					errors[inputIndex].Add(error, errors[inputIndex]);
				}
			}
			
			return errors;
		}

		public void Gradient (InputT[] input, OutputT[] outputError)
		{
			if (input.Length * layerCount != outputError.Length || outputError.Length != outputCount)
				throw new ArgumentException("Count of layers do not match.");
			
			for (var inputIndex = 0; inputIndex < inputCount; ++inputIndex)
				for (var layerIndex = 0; layerIndex < layerCount; ++layerIndex)
				{
					var errorIndex = inputIndex + layerIndex * inputCount;
					layers[layerIndex].Gradient(input[inputIndex], outputError[errorIndex]);
				}
		}

		public void Correct (double coeff)
		{
			foreach(var layer in layers)
				layer.Correct(coeff);
		}

		public OutputT[] Output {
			get {
				return outputs;
			}
		}
		#endregion
	}
}

