using System.Collections.Generic;
using System.Linq;

using Nanon.Math.Linear;
using Nanon.NeuralNetworks.Layer;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public class CompositeLayer<InputT, HiddenT, OutputT>: ICompositeLayer<InputT, OutputT>
	{
		ISingleLayer<InputT,  HiddenT> first;
		ICompositeLayer<HiddenT, OutputT> second;
		
		public CompositeLayer(ISingleLayer<InputT, HiddenT> fst, ICompositeLayer<HiddenT, OutputT> snd)
		{
			first  = fst;
			second = snd;
		}
		
		#region ICompositeLayer[InputT,OutputT] implementation
		
		public OutputT FeedForward(InputT input)
		{
			var hidden = first.FeedForward(input);
			var output = second.FeedForward(hidden);
			return output;
		}

		public InputT PropagateBackward(InputT input, InputT signal, OutputT error)
		{
			var errorInHiddenLayer = second.PropagateBackward(first.Output, first.Signal, error);
			first.FindGradient(input, errorInHiddenLayer);
			return first.PropagateBackward(input, signal, errorInHiddenLayer);
		}
		
		public void Backprop(InputT input, OutputT error)
		{
			var errorInHiddenLayer = second.PropagateBackward(first.Output, first.Signal, error);
			first.FindGradient(input, errorInHiddenLayer);
		}

		public ConsList<Vector> Gradient()
		{
			var fgrad = first.Gradient();
			var restGrads = second.Gradient();
			
			return ConsList<Vector>.Cons(fgrad, restGrads);
		}

		public void Correct(ConsList<Vector> gradients)
		{
			first.Correct(gradients.Head);
			second.Correct(gradients.Tail);
		}

		public ConsList<Vector> ZeroGradients()
		{
			var fgrad = first.ZeroGradients();
			var restGrads = second.ZeroGradients();
			
			return ConsList<Vector>.Cons(fgrad, restGrads);
		}

		public OutputT Signal {
			get {
				throw new System.NotImplementedException ();
			}
		}
		
		#endregion
	}
}

