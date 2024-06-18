package com.griddynamics.capstone_project.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;

import org.springframework.http.HttpStatusCode;
import org.springframework.lang.NonNull;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.reactive.function.client.WebClientException;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import reactor.core.publisher.Mono;

@Service
//@PropertySource("classpath:auth.properties")
public class AudioAuthService {
    private final WebClient webClient;

    public AudioAuthService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl("http://localhost:8000").build();
    }

    public String predictUser(@NonNull MultipartFile file) throws Exception {
        try {
            return webClient.post()
                    .uri("/predict/")
                    .body(BodyInserters.fromMultipartData("file", file.getResource()))
                    .retrieve()
                    .onStatus(HttpStatusCode::is4xxClientError, response -> Mono.error(new RuntimeException("Client error: " + response.statusCode())))
                    .onStatus(HttpStatusCode::is5xxServerError, response -> Mono.error(new RuntimeException("Server error: " + response.statusCode())))
                    .bodyToMono(String.class)
                    .block();
        } catch (WebClientResponseException e) {
            // Handle specific WebClient response exceptions
            throw new RuntimeException("WebClient response error: " + e.getStatusCode() + " " + e.getResponseBodyAsString(), e);
        } catch (WebClientException e) {
            // Handle other WebClient exceptions
            throw new RuntimeException("WebClient error: " + e.getMessage(), e);
        } catch (Exception e) {
            // Handle unexpected exceptions
            throw new RuntimeException("Unexpected error: " + e.getMessage(), e);
        }
    }
}
