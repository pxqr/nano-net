using System;
using Nanon.NeuralNetworks.Layer;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public class Applicator<InputT, OutputT> : ISingleLayer<InputT[], OutputT[]>
	{
		ISingleLayer<InputT, OutputT>[] layers;
		OutputT[] outputs;
		InputT[]  errors;
		int size;
		
		public Applicator(ISingleLayer<InputT, OutputT>[] parLayers)
		{
			layers = parLayers;
			size   = layers.Length;
			outputs = new OutputT[size];
			errors  = new InputT[size];
		}

		#region ISingleLayer[InputT[],OutputT[]] implementation
		
		public OutputT[] FeedForward (InputT[] input)
		{
			for (var i = 0; i < size; ++i)
			{
				outputs[i] = layers[i].FeedForward(input[i]);
			}
			return outputs;
		}

		public InputT[] PropagateBackward (InputT[] input, OutputT[] error)
		{
			for (var i = 0; i < size; ++i)
			{
				errors[i] = layers[i].PropagateBackward(input[i], error[i]);
			}
			return errors;
		}

		public void Gradient (InputT[] input, OutputT[] outputError)
		{
			for (var i = 0; i < size; ++i)
				layers[i].Gradient(input[i], outputError[i]);
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

