using System;
using System.Collections.Generic;

using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer
{
	enum LayerType { FullyConnected, Convolutional };
	
	// Layer :: input -> output
	public interface ISingleLayer<InputT, OutputT>
	{
		OutputT FeedForward(InputT input);
		
		InputT  PropagateBackward(InputT input, InputT predSignal, OutputT error);
		OutputT Signal { get; }
		OutputT Output { get; }
		
		void FindGradient(InputT input, OutputT outputError);
		Vector Gradient();		
		void Correct(Vector gradients);
		Vector ZeroGradients();
		
		// weights
		//double this [int i] { get; set; }
		//int WeightsSize  { get; } 
	}
}