using System;

namespace Nanon.NeuralNetworks.Layer.Composition
{
	public class ConsList<T>
	{
		readonly T head;
		readonly ConsList<T> tail;
		
		public ConsList(T head, ConsList<T> tail)
		{
			this.head = head;
			this.tail = tail;	
		}
		
		public T Head {
			get {
				return this.head;
			}
		}

		public ConsList<T> Tail {
			get {
				return this.tail;
			}
		}
		
		public static ConsList<T> Cons(T head, ConsList<T> tail)
		{
			return new ConsList<T>(head, tail);
		}
		
		static A Foldr<A, B>(Func<B, A, A> f, A seed, B[] arr)
		{
			var acc = seed;
			for (var i = arr.Length - 1; i >= 0; --i)
				acc = f(arr[i], acc);
			return acc;
		}
		
		public static ConsList<T> FromArray(T[] arr)
		{
			return Foldr<ConsList<T>, T>(Cons, null, arr);
		}
		
		public int Length
		{
			get
			{
				if (tail == null)
					 return 1;
				else return 1 + tail.Length;
			}
		}
		
		public T[] ToArray()
		{
			var len = Length;
			var arr = new T[len];
			
			var curr = this;
			var i = 0;
			
			while (curr != null)
			{
				arr[i] = curr.head;
				++i;
				curr = curr.tail;
			}
			
			return arr;
		}
	}
}

