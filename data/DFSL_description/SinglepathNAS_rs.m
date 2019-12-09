Network SinglepathNAS {
Layer Conv-1 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 32, C: 3, R: 3, S: 3, Y: 224, X: 224 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 32, R: 1, S: 1, Y: 112, X: 112 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-3 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 96, R: 3, S: 3, Y: 112, X: 112 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 16, C: 96, R: 1, S: 1, Y: 56, X: 56 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-5 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 48, C: 16, R: 1, S: 1, Y: 56, X: 56 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-6 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 48, R: 3, S: 3, Y: 56, X: 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-7 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 48, R: 1, S: 1, Y: 56, X: 56 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-8 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 72, C: 24, R: 1, S: 1, Y: 56, X: 56 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-9 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 72, R: 3, S: 3, Y: 56, X: 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-10 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 72, R: 1, S: 1, Y: 56, X: 56 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-11 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 144, C: 24, R: 1, S: 1, Y: 56, X: 56 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-12 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 144, R: 5, S: 5, Y: 56, X: 56 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-13 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 24, C: 144, R: 1, S: 1, Y: 28, X: 28 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-14 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 72, C: 24, R: 1, S: 1, Y: 28, X: 28 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-15 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 72, R: 3, S: 3, Y: 28, X: 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-16 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 40, C: 72, R: 1, S: 1, Y: 28, X: 28 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-17 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 120, C: 40, R: 1, S: 1, Y: 28, X: 28 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-18 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 120, R: 3, S: 3, Y: 28, X: 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-19 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 40, C: 120, R: 1, S: 1, Y: 28, X: 28 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-20 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 120, C: 40, R: 1, S: 1, Y: 28, X: 28 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-21 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 120, R: 3, S: 3, Y: 28, X: 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-22 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 40, C: 120, R: 1, S: 1, Y: 28, X: 28 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-23 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 240, C: 40, R: 1, S: 1, Y: 28, X: 28 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-24 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 240, R: 5, S: 5, Y: 28, X: 28 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-25 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 240, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-26 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 240, C: 80, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-27 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 240, R: 3, S: 3, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-28 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 240, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-29 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 240, C: 80, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-30 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 240, R: 3, S: 3, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-31 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 240, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-32 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 240, C: 80, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-33 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 240, R: 3, S: 3, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-34 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 80, C: 240, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-35 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 480, C: 80, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-36 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 480, R: 5, S: 5, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-37 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 480, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-38 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 288, C: 96, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-39 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 288, R: 5, S: 5, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-40 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 288, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-41 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 288, C: 96, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-42 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 288, R: 5, S: 5, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-43 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 288, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-44 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 288, C: 96, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-45 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 288, R: 5, S: 5, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-46 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 96, C: 288, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-47 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 576, C: 96, R: 1, S: 1, Y: 14, X: 14 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-48 {
Type: DSCONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1, C: 576, R: 5, S: 5, Y: 14, X: 14 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-49 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 576, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-50 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1152, C: 192, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-51 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 1152, R: 5, S: 5, Y: 7, X: 7 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-52 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 1152, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-53 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1152, C: 192, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-54 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 1152, R: 5, S: 5, Y: 7, X: 7 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-55 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 1152, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-56 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1152, C: 192, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-57 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 1152, R: 5, S: 5, Y: 7, X: 7 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-58 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 1152, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-59 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1152, C: 192, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-60 {
Type: DSCONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1, C: 1152, R: 3, S: 3, Y: 7, X: 7 }
Dataflow {
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
}
Layer Conv-61 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 320, C: 1152, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Conv-62 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1280, C: 320, R: 1, S: 1, Y: 7, X: 7 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
Layer Linear-63 {
Type: CONV
Dimensions { K: 1000, C: 1280, R: 1, S: 1, Y: 1, X: 1 }
 Dataflow {
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
}
}