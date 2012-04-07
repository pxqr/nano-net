using System;
using System.Collections.Generic;

using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public interface ICompositeLayer<InputT, OutputT>
	{
		OutputT FeedForward(InputT input);
		
		void Backprop(InputT input, OutputT error);
		InputT  PropagateBackward(InputT input, InputT signal, OutputT error);
		
		ConsList<Vector> Gradient();		
		void   Correct(ConsList<Vector> gradients);
		ConsList<Vector> ZeroGradients();
	}
}

