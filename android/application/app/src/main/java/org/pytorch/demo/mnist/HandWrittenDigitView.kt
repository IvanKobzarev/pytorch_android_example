// CoMutableListght (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


package org.pytorch.demo.mnist;

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.util.Pair
import android.view.MotionEvent
import android.view.View
import java.util.Collections.emptyList

class HandWrittenDigitView : View {
    private var mPath: Path? = null
    private var mPaint: Paint? = null
    private var mCanvasPaint: Paint? = null;

    private var mCanvas: Canvas? = null
    private var mBitmap: Bitmap? = null
    private var mAllPoints: MutableList<Pair<Float, Float>> = emptyList()
    private var mConsecutivePoints: MutableList<Pair<Float, Float>> = emptyList()

    constructor(context: Context, attrs: AttributeSet, defStyle: Int) : super(context, attrs, defStyle) {}
    constructor(context: Context, attrs: AttributeSet) : this(context, attrs, 0) {}

    private fun setPathPaint() {
        mPath = Path()
        mPaint = Paint()
        mPaint!!.color = 0xFF000000.toInt()
        mPaint!!.isAntiAlias = true
        mPaint!!.strokeWidth = 18f
        mPaint!!.style = Paint.Style.STROKE
        mPaint!!.strokeJoin = Paint.Join.ROUND
        mCanvasPaint = Paint(Paint.DITHER_FLAG)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh);
        mBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        mCanvas = Canvas(mBitmap)
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawBitmap(mBitmap, 0f, 0f, mCanvasPaint)
        canvas.drawPath(mPath, mPaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        var x = event.x
        var y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                mConsecutivePoints.clear()
                mConsecutivePoints.add(Pair(x, y))
                mPath!!.moveTo(x, y)
            }
            MotionEvent.ACTION_MOVE -> {
                mConsecutivePoints.add(Pair(x, y))
                mPath!!.lineTo(x, y);
            }
            MotionEvent.ACTION_UP -> {
                mConsecutivePoints.add(Pair(x, y))
                mAllPoints.addAll(mConsecutivePoints)
                mCanvas?.drawPath(mPath, mPaint)
                mPath?.reset()
            }
            else -> return false;
        }
        invalidate()
        return true
    }

    fun getAllPoints(): MutableList<Pair<Float, Float>> {
        return mAllPoints;
    }

    fun clearAllPointsAndRedraw() {
        mBitmap = Bitmap.createBitmap(mBitmap!!.width, mBitmap!!.height, Bitmap.Config.ARGB_8888);
        mCanvas = Canvas(mBitmap)
        mCanvasPaint = Paint(Paint.DITHER_FLAG);
        mCanvas!!.drawBitmap(mBitmap, 0f, 0f, mCanvasPaint);

        setPathPaint()
        invalidate()
        mAllPoints.clear()
    }

    fun clearAllPoints() {
        mAllPoints.clear()
    }
}