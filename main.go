package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api"
	"github.com/go-resty/resty/v2"
)

type Recipe struct {
	Title       string `json:"title"`
	Ingredients string `json:"ingredients"`
	Steps       string `json:"steps"`
}

func main() {
	token := "7301752884:AAHCSJPakGKqcktQFfwAuo1gISmqtKb9j7E"
	bot, err := tgbotapi.NewBotAPI(token)
	if err != nil {
		log.Panic(err)
	}

	bot.Debug = true
	u := tgbotapi.NewUpdate(0)
	u.Timeout = 60

	updates, _ := bot.GetUpdatesChan(u)

	client := resty.New()

	for update := range updates {
		if update.Message == nil {
			continue
		}

		if update.Message.IsCommand() {
			switch update.Message.Command() {
			case "start":
				msg := tgbotapi.NewMessage(update.Message.Chat.ID, "Selamat datang di chatbot rekomendasi resep masakan Indonesia. Silahkan masukkan bahan-bahan yang Anda miliki.")
				bot.Send(msg)
				continue
			}
		}

		ingredients := update.Message.Text

		resp, err := client.R().
			SetHeader("Content-Type", "application/json").
			SetBody(map[string]interface{}{"ingredients": ingredients}).
			Post("http://localhost:5000/recommend")

		if err != nil {
			log.Println(err)
			msg := tgbotapi.NewMessage(update.Message.Chat.ID, "Terjadi kesalahan dalam menghubungi server.")
			bot.Send(msg)
			continue
		}

		var recommendations []Recipe
		err = json.Unmarshal(resp.Body(), &recommendations)
		if err != nil {
			log.Println(err)
			msg := tgbotapi.NewMessage(update.Message.Chat.ID, "Terjadi kesalahan dalam memproses respon dari server.")
			bot.Send(msg)
			continue
		}

		msg := tgbotapi.NewMessage(update.Message.Chat.ID, formatRecommendations(recommendations))
		bot.Send(msg)
	}
}

func formatRecommendations(recipes []Recipe) string {
	if len(recipes) == 0 {
		return "Tidak ada resep yang cocok dengan bahan-bahan yang Anda masukkan."
	}

	var result strings.Builder
	result.WriteString("Berikut Adalah Rekomendasi resep Yang Mirip Berdasarkan Bahan Bahan Anda:\n")
	for _, recipe := range recipes {
		result.WriteString(fmt.Sprintf("\nNama Resep: %s\nBahan-bahan: %s\nLangkah-langkah: %s\n", recipe.Title, recipe.Ingredients, recipe.Steps))
	}
	return result.String()
}