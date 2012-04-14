using System;
using System.Linq;

using Nanon.NeuralNetworks.Layer;
using Nanon.Math.Linear;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public class LayerCompositor
	{
		/*
		// Foldl1 :: Layer a => [a] -> (a -> a -> a) -> a
		public static ICompositeLayer<T, T> Foldl1<T>(ISingleLayer<T, T>[] layers, 
		                                              Func<ISingleLayer<T, T>, ICompositeLayer<T, T>, ICompositeLayer<T, T>> f)
		{
			if (layers.Length < 1)
				throw new Exception("Foldl1 isnt defined for empty container.");
			
			var acc = layers[0];
			for (int i = 1; i < layers.Length; ++i)
				acc = f(acc, layers[i]);
			     
			return acc;
		} 
		*/
		// Compose :: Layer (a -> b) -> Layer (b -> c) -> Layer (a -> c)
	  	public static ICompositeLayer<A, C> Compose<A, B, C>(ISingleLayer<A, B> first, ICompositeLayer<B, C> second)
	  	{
			return new CompositeLayer<A, B, C>(first, second);
		}
		
		public static ICompositeLayer<A, A> ComposeGeteroneneous<A>(ISingleLayer<A, A>[] layers)
			where A : IMatrix<A>
		{
			ICompositeLayer<A, A> acc = new OutputLayer<A>();
			return layers.Reverse().Aggregate(acc, (a, b) => Compose(b, a));
		}
	}
}

