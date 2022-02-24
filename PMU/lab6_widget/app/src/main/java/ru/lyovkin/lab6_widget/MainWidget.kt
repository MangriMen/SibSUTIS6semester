package ru.lyovkin.lab6_widget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.Context
import android.content.Intent
import android.widget.RemoteViews

/**
 * Implementation of App Widget functionality.
 */
class MainWidget : AppWidgetProvider() {
    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray
    ) {
        // There may be multiple widgets active, so update all of them
        for (appWidgetId in appWidgetIds) {
            val intent = Intent(context, MainActivity::class.java)
            val pendingIntent = PendingIntent.getActivity(context, 0, intent, 0)

            val views = RemoteViews(context.packageName, R.layout.main_widget)
            views.setOnClickPendingIntent(R.id.appwidget_text, pendingIntent)

            updateAppWidget(context, appWidgetManager, appWidgetId)
            appWidgetManager.updateAppWidget(appWidgetId, views);
        }
    }

    override fun onEnabled(context: Context) {
        // Enter relevant functionality for when the first widget is created
    }

    override fun onDisabled(context: Context) {
        // Enter relevant functionality for when the last widget is disabled
    }
}

internal fun updateAppWidget(
    context: Context,
    appWidgetManager: AppWidgetManager,
    appWidgetId: Int
) {
//    val widgetText = context.getString(R.string.appwidget_text)
    val data = MainActivity.getData()
    val widgetText = "Дата: ${data["Дата"]}\nЗолото: ${data["Золото"]}\n"

    // Construct the RemoteViews object
    val views = RemoteViews(context.packageName, R.layout.main_widget)
    views.setTextViewText(R.id.appwidget_text, widgetText)

    // Instruct the widget manager to update the widget
    appWidgetManager.updateAppWidget(appWidgetId, views)
}