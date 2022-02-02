package ru.lyovkin.lab1_calculator;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.res.ResourcesCompat;

import java.math.BigDecimal;
import java.math.MathContext;

public class MainActivity extends AppCompatActivity {
    enum Action {
        None,
        Plus,
        Minus,
        Mul,
        Div,
    }

    private TextView tvNumber;
    private TextView tvBuffer;
    private Action currentAction = Action.None;
    private boolean hasDot = false;
    private boolean isNewNumber = false;
    final private int PRECISION = 12;
    final private MathContext PRECISION_CONTEXT = new MathContext(PRECISION);

    Button oldButton;

    private int standardBackground;
    private int standardForeground;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvNumber = findViewById(R.id.tvNumber);
        tvBuffer = findViewById(R.id.tvBuffer);

        clearTextViewByZero(tvNumber);

        standardBackground = ResourcesCompat.getColor(getResources(), R.color.white, null);
        standardForeground = ResourcesCompat.getColor(getResources(), R.color.purple_500, null);
    }

    public void clearTextView(@NonNull TextView textView) {
        final String empty = "";
        textView.setText(empty);
    }

    public void clearTextViewByZero(@NonNull TextView textView) {
        final String zero = "0";
        textView.setText(zero);
    }

    public String dropLast(@NonNull String str, int n) {
        if (str.length() - n < 0) {
            return "";
        }

        return str.substring(0, str.length() - n);
    }

    public void setButtonStyleActive(@NonNull Button button) {
        button.setBackgroundColor(standardForeground);
        button.setTextColor(standardBackground);
    }

    public void setButtonStyleDefault(@NonNull Button button) {
        button.setBackgroundColor(standardBackground);
        button.setTextColor(standardForeground);
    }

    public void onClearClick(View view) {
        clearTextViewByZero(tvNumber);
        clearTextView(tvBuffer);
        hasDot = false;
        currentAction = Action.None;

        setButtonStyleDefault(findViewById(R.id.btnPlus));
        setButtonStyleDefault(findViewById(R.id.btnMinus));
        setButtonStyleDefault(findViewById(R.id.btnMul));
        setButtonStyleDefault(findViewById(R.id.btnDiv));
    }

    public void onRemoveClick(View view) {
        if (tvNumber.getText().toString().equals("0")) {
            return;
        }

        if (tvNumber.getText().charAt(tvNumber.getText().length() - 1) == '.') {
            hasDot = false;
        }

        tvNumber.setText(dropLast(tvNumber.getText().toString(), 1));

        if (tvNumber.getText().length() == 0) {
            clearTextViewByZero(tvNumber);
        }
    }

    public void onDigitClick(View view) {
        if (tvNumber.getText().toString().equals("0")) {
            clearTextView(tvNumber);
        }

        if (isNewNumber) {
            clearTextView(tvNumber);
            isNewNumber = false;
        }

        Button btnDigit = (Button)view;

        tvNumber.append(btnDigit.getText());
    }

    public void onDotClick(View view) {
        if (hasDot) {
            return;
        }

        hasDot = true;
        tvNumber.append(".");
    }

    public void onActionClick(View view) {
        final Button btnAction = (Button)view;
        final String btnText = btnAction.getText().toString();

        if (currentAction == Action.None) {
            if (tvNumber.getText().toString().equals(getResources().getString(R.string._error))) {
                clearTextViewByZero(tvNumber);
            }

            tvBuffer.setText(tvNumber.getText());
            clearTextViewByZero(tvNumber);
            hasDot = false;

            if (btnText.equals(getResources().getString(R.string._plus))) {
                currentAction = Action.Plus;
            }
            else if (btnText.equals(getResources().getString(R.string._minus))) {
                currentAction = Action.Minus;
            }
            else if (btnText.equals(getResources().getString(R.string._mul))) {
                currentAction = Action.Mul;
            }
            else if (btnText.equals(getResources().getString(R.string._div))) {
                currentAction = Action.Div;
            }

            oldButton = btnAction;

            setButtonStyleActive(btnAction);

            return;
        }

        if (btnText.equals(getResources().getString(R.string._equal))) {
            BigDecimal firstNumber = new BigDecimal(tvBuffer.getText().toString());
            BigDecimal secondNumber = new BigDecimal(tvNumber.getText().toString());
            BigDecimal result = null;

            switch (currentAction) {
                case Plus:
                    result = firstNumber.add(secondNumber, PRECISION_CONTEXT);
                    break;
                case Minus:
                    result = firstNumber.subtract(secondNumber, PRECISION_CONTEXT);
                    break;
                case Mul:
                    result = firstNumber.multiply(secondNumber, PRECISION_CONTEXT);
                    break;
                case Div:
                    if (secondNumber.equals(new BigDecimal(0))) {
                        tvNumber.setText(getResources().getString(R.string._error));
                        clearTextView(tvBuffer);

                        currentAction = Action.None;
                        isNewNumber = true;
                        setButtonStyleDefault(oldButton);
                        return;
                    }
                    else {
                        result = firstNumber.divide(secondNumber, PRECISION_CONTEXT);
                    }
                    break;
            }

            tvNumber.setText(String.valueOf(result));
            clearTextView(tvBuffer);

            currentAction = Action.None;
            isNewNumber = true;

            setButtonStyleDefault(oldButton);
        }
    }
}